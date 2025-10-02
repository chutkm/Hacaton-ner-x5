"""
Главный скрипт тренировки.

Поддерживает:
- stage1: head-only training (freeze base)
- stage2: full fine-tune (unfreeze) с сохранением чекпоинтов (раз в эпоху)
- загрузку отдельных частей модели перед обучением (--load_parts_dir)
- сохранение run-папки (--save_parts)
- random-threshold search на валидации
- Optuna hyperparameter search (опционально)
"""
import argparse
import os
import random
import json
import shutil
import logging
from typing import List, Dict, Any
from collections import Counter

import numpy as np
import torch
import optuna
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification, AutoTokenizer
from src.dataset import load_hf_datasets
from src.model.modular import ModularTokenClassifier
from src.metrics import macro_f1, parse_annotation
from src.utils import (
    logits_to_labels_and_confs,
    make_run_dir,
    load_model_parts,
    compute_class_weights_from_counts,
    LABELS,
    fix_bio_labels
)
from datasets import Dataset

optuna.logging.set_verbosity(optuna.logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_training_args(base_kwargs: dict) -> TrainingArguments:
    """
    Создаёт TrainingArguments, подставляя правильные имена аргументов
    в зависимости от конкретной версии transformers, установленной в окружении.
    """
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    mapped = {}
    for k, v in base_kwargs.items():
        if k in params:
            mapped[k] = v

    if "eval_strategy" in base_kwargs and "eval_strategy" not in mapped:
        if "eval_strategy" in params:
            mapped["eval_strategy"] = base_kwargs["eval_strategy"]

    if "logging_strategy" in base_kwargs and "logging_strategy" not in mapped:
        if "logging_strategy" in params:
            mapped["logging_strategy"] = base_kwargs["logging_strategy"]

    if "save_strategy" in base_kwargs and "save_strategy" not in mapped:
        if "save_strategy" in params:
            mapped["save_strategy"] = base_kwargs["save_strategy"]
        elif "save_steps" in params and isinstance(base_kwargs["save_strategy"], (int, float)):
            mapped["save_steps"] = base_kwargs["save_strategy"]

    if "output_dir" in base_kwargs and "output_dir" not in mapped and "output_dir" in params:
        mapped["output_dir"] = base_kwargs["output_dir"]

    for k, v in base_kwargs.items():
        if k not in mapped and k in params:
            mapped[k] = v

    return TrainingArguments(**mapped)


def gather_val_predictions(model, tokenizer, val_examples: List[Dict[str,str]], device: str, max_length: int = 128):
    """
    Прогоняет модель по листу примеров (sample, annotation) и возвращает
    структуру с токеновыми offset'ами, предсказанными метками и confidence.
    """
    model.to(device)
    model.eval()
    results = []
    for ex in val_examples:
        text = ex["sample"]
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_length)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)

        logits = out.get("logits", None)
        if logits is None:
            results.append({"sample": text, "offsets": [], "pred_labels": [], "token_confs": [], "annotation": ex["annotation"]})
            continue

        logits = logits[0].cpu()
        preds_field = out.get("predictions", None)

        if preds_field is not None:
            if isinstance(preds_field, list):
                pred_ids = preds_field[0]
            else:
                pred_ids = preds_field[0].tolist()
            token_labels = [LABELS[i] if 0 <= i < len(LABELS) else "O" for i in pred_ids]
            _, token_confs = logits_to_labels_and_confs(logits, {i: l for i, l in enumerate(LABELS)})
        else:
            token_labels, token_confs = logits_to_labels_and_confs(logits, {i: l for i, l in enumerate(LABELS)})

        token_offsets = []
        token_label_list = []
        token_conf_list = []
        for (a, b), lab, conf in zip(offsets, token_labels, token_confs):
            if a == b:
                continue
            token_offsets.append((a, b))
            token_label_list.append(lab)
            token_conf_list.append(conf)

        token_label_list = fix_bio_labels(token_label_list)

        results.append({
            "sample": text,
            "offsets": token_offsets,
            "pred_labels": token_label_list,
            "token_confs": token_conf_list,
            "annotation": ex["annotation"]
        })
    return results


def random_threshold_search(val_preds: List[Dict[str,Any]], n_iter: int = 50, seed: int = 42) -> Dict[str, Any]:
    rnd = random.Random(seed)
    best = {"thr": 0.5, "score": -1.0}

    for _ in range(n_iter):
        thr = rnd.uniform(0.1, 0.95)
        true_chunks = []
        pred_chunks = []

        for r in val_preds:
            true = parse_annotation(r.get("annotation", "[]"))
            spans = []
            current_span = None
            current_confs = []

            offsets = r.get("offsets", [])
            pred_labels = r.get("pred_labels", [])
            token_confs = r.get("token_confs", [])

            # Fix BIO in pred_labels BEFORE converting to spans
            pred_labels = fix_bio_labels(pred_labels)

            L = min(len(offsets), len(pred_labels), len(token_confs) if token_confs else len(pred_labels))
            if not token_confs:
                token_confs = [1.0] * len(pred_labels)

            for i in range(L):
                (a, b) = offsets[i]
                lab = pred_labels[i]
                conf = float(token_confs[i])

                if a == b:
                    if current_span:
                        spans.append({
                            "start_index": current_span["start_index"],
                            "end_index": current_span["end_index"],
                            "entity": current_span["entity"],
                            "original_label": current_span["original_label"],
                            "mean_conf": np.mean(current_confs) if current_confs else 0.0
                        })
                        current_span = None
                        current_confs = []
                    continue

                if lab == "O" or "-" not in lab:
                    if current_span:
                        spans.append({
                            "start_index": current_span["start_index"],
                            "end_index": current_span["end_index"],
                            "entity": current_span["entity"],
                            "original_label": current_span["original_label"],
                            "mean_conf": np.mean(current_confs) if current_confs else 0.0
                        })
                        current_span = None
                        current_confs = []
                    continue

                prefix, entity = lab.split("-", 1)

                if prefix == "B" or current_span is None or current_span["entity"] != entity:
                    if current_span:
                        spans.append({
                            "start_index": current_span["start_index"],
                            "end_index": current_span["end_index"],
                            "entity": current_span["entity"],
                            "original_label": current_span["original_label"],
                            "mean_conf": np.mean(current_confs) if current_confs else 0.0
                        })
                    current_span = {
                        "start_index": a,
                        "end_index": b,
                        "entity": entity,
                        "original_label": lab
                    }
                    current_confs = [conf]
                elif prefix == "I" and current_span and current_span["entity"] == entity:
                    current_span["end_index"] = b
                    current_confs.append(conf)

            if current_span:
                spans.append({
                    "start_index": current_span["start_index"],
                    "end_index": current_span["end_index"],
                    "entity": current_span["entity"],
                    "original_label": current_span["original_label"],
                    "mean_conf": np.mean(current_confs) if current_confs else 0.0
                })

            filtered_spans = [
                (span["start_index"], span["end_index"], span["original_label"])
                for span in spans
                if span["mean_conf"] >= thr
            ]

            pred_chunks.append(filtered_spans)
            true_chunks.append(true)

        score = macro_f1(true_chunks, pred_chunks)
        if score > best["score"]:
            best = {"thr": thr, "score": score}

    return best


def build_class_weights_from_dataset(train_dataset, min_w: float = 0.2, max_w: float = 10.0, normalize_mean: bool = True):
    """
    train_dataset: HF Dataset, колонка 'labels' содержит списки int (или -100)
    Вернёт (weights_tensor, counts)
    """
    counter = Counter()
    for ex in train_dataset:
        labs = ex.get("labels", [])
        if labs is None:
            continue
        for lab in labs:
            if lab == -100:
                continue
            if 0 <= int(lab) < len(LABELS):
                counter[LABELS[int(lab)]] += 1

    counts = {lab: int(counter.get(lab, 0)) for lab in LABELS}

    if sum(counts.values()) == 0:
        print("Warning: train dataset label counts are zero; using uniform weights.")
        return torch.ones(len(LABELS), dtype=torch.float), counts

    raw_weights = compute_class_weights_from_counts(counts).numpy()
    if normalize_mean:
        raw_weights = raw_weights / float(raw_weights.mean())

    raw_weights = np.clip(raw_weights, a_min=min_w, a_max=max_w)
    weights_tensor = torch.tensor(raw_weights, dtype=torch.float)
    return weights_tensor, counts


def optuna_objective(trial, args, tokenizer, train_ds, val_ds, val_examples, device, run_dir, class_weights):
    """
    Optuna objective: создаёт модель с предлагаемыми trial гиперпараметрами,
    тренирует stage1/stage2 (если включены) и возвращает F1 на валидации.
    """
    # Гиперпараметры для оптимизации
    stage1_lr = trial.suggest_loguniform("stage1_lr", 1e-5, 5e-4)
    stage2_lr = trial.suggest_loguniform("stage2_lr", 1e-6, 1e-4)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    train_bs = int(trial.suggest_categorical("train_bs", [4, 8, 16]))

    # добавим поиск ce_weight (коэффициент CE loss при использовании CRF)
    ce_weight = trial.suggest_float("ce_weight", 0.1, 1.0)

    set_seed(args.seed + trial.number)

    s1_epochs = min(1, args.stage1_epochs) if args.optuna_fast else args.stage1_epochs
    s2_epochs = min(3, args.stage2_epochs) if args.optuna_fast else args.stage2_epochs

    model = ModularTokenClassifier.from_pretrained_base(
        args.model_name_or_path,
        use_crf=args.use_crf,
        num_labels=len(LABELS),
        extra_dropout=dropout,
        class_weights=class_weights,
        ce_weight=ce_weight
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    trial_dir = os.path.join(run_dir, f"optuna_trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    try:
        if args.stage1_head_only:
            model.freeze_base()
            base_kwargs = {
                "output_dir": os.path.join(trial_dir, "stage1_ckpt"),
                "num_train_epochs": s1_epochs,
                "per_device_train_batch_size": train_bs,
                "per_device_eval_batch_size": args.eval_bs,
                "learning_rate": stage1_lr,
                "eval_strategy": "epoch",
                "save_strategy": "no",
                "logging_steps": 50,
                "fp16": args.fp16
            }
            training_args = create_training_args(base_kwargs)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            trainer.train()

            del trainer
            torch.cuda.empty_cache()

        if args.stage2_full:
            model.unfreeze_base()
            base_kwargs = {
                "output_dir": os.path.join(trial_dir, "stage2_ckpt"),
                "num_train_epochs": s2_epochs,
                "per_device_train_batch_size": train_bs,
                "per_device_eval_batch_size": args.eval_bs,
                "learning_rate": stage2_lr,
                "eval_strategy": "epoch",
                "save_strategy": "no",
                "logging_steps": 50,
                "fp16": args.fp16
            }
            training_args = create_training_args(base_kwargs)

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            trainer.train()

            del trainer
            torch.cuda.empty_cache()

        val_preds = gather_val_predictions(model, tokenizer, val_examples, device, max_length=args.max_length)
        best_thr = random_threshold_search(val_preds, n_iter=args.thr_search_iter, seed=args.seed)
        score = best_thr["score"]

        trial.report(score, 0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()

    finally:
        try:
            shutil.rmtree(trial_dir)
        except Exception:
            pass
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass


manual_counts = {
  "O": 5379,
  "B-TYPE": 24528,
  "I-TYPE": 4532,
  "B-BRAND": 7212,
  "I-BRAND": 487,
  "B-VOLUME": 57,
  "I-VOLUME": 27,
  "B-PERCENT": 26,
  "I-PERCENT": 4
}



def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    train_ds, val_ds = load_hf_datasets(args.train_csv, args.val_csv, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    # run dir
    run_dir = make_run_dir(base_dir="runs", run_name=args.run_name)
    print("Run dir:", run_dir)

    # optional load parts (for inspection)
    if args.load_parts_dir:
        parts = [p.strip() for p in args.parts_to_load.split(",") if p.strip()]
        print("Loading parts", parts, "from", args.load_parts_dir)
        placeholder = ModularTokenClassifier.from_pretrained_base(
            args.model_name_or_path,
            use_crf=args.use_crf,
            num_labels=len(LABELS),
            extra_dropout=args.dropout
        )
        loaded = load_model_parts(placeholder, args.load_parts_dir, parts_to_load=parts, device=device)
        print("Loaded parts:", loaded)
        del placeholder

    # Подготовка валидационных примеров
    val_examples = [{"sample": x["sample"], "annotation": x["annotation"]} for x in val_ds]

    # --- Считаем веса по train_ds ---
    print("Computing class weights from training dataset...")
    weights_tensor, counts = build_class_weights_from_dataset(train_ds, min_w=0.2, max_w=10.0)
    # weights_tensor, counts = manual_counts
    print("Token counts:", counts)
    print("Class weights (applied):")
    for i, lab in enumerate(LABELS):
        print(f"  {lab:12s}: {float(weights_tensor[i].item()):6.4f}")

    meta_run_info = {
        "class_weights": {LABELS[i]: float(weights_tensor[i].item()) for i in range(len(LABELS))},
        "class_counts": counts
    }

    # Optuna hyperparameter search (опционально)
    study = None
    if args.optuna_trials > 0:
        print(f"Starting Optuna hyperparameter search ({args.optuna_trials} trials)...")
        print(f"  - Fast mode: {'enabled' if args.optuna_fast else 'disabled'}")
        print(f"  - Pruner: {'enabled' if args.optuna_pruner else 'disabled'}")

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1) if args.optuna_pruner else optuna.pruners.NopPruner()

        study = optuna.create_study(
            direction="maximize",
            study_name=f"ner_study_{os.path.basename(run_dir)}",
            pruner=pruner,
            load_if_exists=True
        )

        try:
            study.optimize(
                lambda trial: optuna_objective(trial, args, tokenizer, train_ds, val_ds, val_examples, device, run_dir, weights_tensor),
                n_trials=args.optuna_trials,
                timeout=args.optuna_timeout,
                gc_after_trial=True,
                show_progress_bar=True
            )

            print("\nOptuna search completed!")
            print(f"Best value: {study.best_value:.4f}")
            print("Best hyperparameters:")
            for param, value in study.best_params.items():
                print(f"  - {param}: {value}")


                if param == "stage1_lr":
                    args.stage1_lr = value
                elif param == "stage2_lr":
                    args.stage2_lr = value
                elif param == "dropout":
                    args.dropout = value
                elif param == "train_bs":
                    args.train_bs = int(value)
                elif param == "ce_weight":
                    args.ce_weight = value

            study.trials_dataframe().to_csv(os.path.join(run_dir, "optuna_results.csv"), index=False)
            with open(os.path.join(run_dir, "optuna_best_params.json"), "w") as f:
                json.dump(study.best_params, f, indent=2)

        except Exception as e:
            print(f"Optuna search failed: {str(e)}")
            print("Falling back to default hyperparameters")

    model = ModularTokenClassifier.from_pretrained_base(
        args.model_name_or_path,
        use_crf=args.use_crf,
        num_labels=len(LABELS),
        extra_dropout=args.dropout,
        class_weights=weights_tensor,
        ce_weight=getattr(args, "ce_weight", 0.7)
    )
    model.to(device)


    if args.load_parts_dir:
        parts = [p.strip() for p in args.parts_to_load.split(",") if p.strip()]
        loaded = load_model_parts(model, args.load_parts_dir, parts_to_load=parts, device=device)
        print("Loaded parts into final model:", loaded)

    # stage1: head-only
    if args.stage1_head_only:
        model.freeze_base()
        print("Stage1: training head only (base frozen).")

        base_kwargs = {
            "output_dir": os.path.join(run_dir, "stage1_ckpt"),
            "num_train_epochs": args.stage1_epochs,
            "per_device_train_batch_size": args.train_bs,
            "per_device_eval_batch_size": args.eval_bs,
            "learning_rate": args.stage1_lr,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "metric_for_best_model": "loss",
            "save_total_limit": 3,
            "fp16": args.fp16
        }
        training_args = create_training_args(base_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(os.path.join(run_dir, "stage1"))

    # stage2: full
    if args.stage2_full:
        model.unfreeze_base()
        print("Stage2: training full model (base unfrozen).")

        base_kwargs = {
            "output_dir": os.path.join(run_dir, "stage2_ckpt"),
            "num_train_epochs": args.stage2_epochs,
            "per_device_train_batch_size": args.train_bs,
            "per_device_eval_batch_size": args.eval_bs,
            "learning_rate": args.stage2_lr,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": 50,
            "load_best_model_at_end": True,
            "metric_for_best_model": "loss",
            "save_total_limit": 3,
            "fp16": args.fp16
        }
        training_args = create_training_args(base_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model(os.path.join(run_dir, "stage2"))

    val_preds = gather_val_predictions(model, tokenizer, val_examples, device, max_length=args.max_length)
    best_thr = random_threshold_search(val_preds, n_iter=args.thr_search_iter, seed=args.seed)
    print("Best threshold found:", best_thr)

    meta = {
        "use_crf": args.use_crf,
        "model_name": args.model_name_or_path,
        "best_threshold": best_thr,
        "train_args": vars(args),
        "optuna_trials": args.optuna_trials,
        "optuna_best_params": study.best_params if args.optuna_trials > 0 and study is not None else None
    }

    meta.update(meta_run_info)

    final_model_dir = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    with open(os.path.join(final_model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Saved final model to", final_model_dir)

    if args.save_parts:
        model.save_run(run_dir, tokenizer=tokenizer, meta=meta)
        with open(os.path.join(run_dir, "train_run_args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        print("Saved run to", run_dir)

# ai-forever/ruRoberta-large
# bert-base-multilingual-cased
#DeepPavlov/bert-base-bg-cs-pl-ru-cased
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train NER model for product search queries")

    # Основные параметры
    p.add_argument("--model_name_or_path", default="bert-base-multilingual-cased",
                   help="Pretrained model name or path")
    p.add_argument("--train_csv", default="/home/ubuntu/x5-ner-bilstm/data/train_fixed_prepos_dots_100_split.csv",
                   help="Path to training CSV")
    p.add_argument("--val_csv", default="/home/ubuntu/x5-ner-bilstm/data/val_fixed_prepos_100_split.csv",
                   help="Path to validation CSV")
    p.add_argument("--output_dir", default="models/v_prep_up",
                   help="Directory to save the final model")
    p.add_argument("--max_length", type=int, default=128,
                   help="Maximum sequence length for tokenizer")

    # Параметры Optuna
    p.add_argument("--optuna_trials", type=int, default=0,
                   help="Number of Optuna trials for hyperparameter search (0 to disable)")
    p.add_argument("--optuna_fast", action="store_true",
                   help="Use reduced epochs in Optuna trials for faster search")
    p.add_argument("--optuna_timeout", type=int, default=None,
                   help="Optuna search timeout in seconds")
    p.add_argument("--optuna_pruner", action="store_true",
                   help="Enable MedianPruner for Optuna to prune unpromising trials")

    # CRF параметры
    p.add_argument("--use_crf", dest="use_crf", action="store_true",
                   help="Enable CRF layer")
    p.add_argument("--no-use_crf", dest="use_crf", action="store_false",
                   help="Disable CRF layer")

    p.add_argument("--dropout", type=float, default=0.22,
                   help="Dropout probability for classifier")

    # Stage1 параметры
    p.add_argument("--stage1_head_only", dest="stage1_head_only", action="store_true",
                   help="Enable stage1: train only head (freeze base)")
    p.add_argument("--no-stage1_head_only", dest="stage1_head_only", action="store_false",
    
                   help="Disable stage1 training")

    p.add_argument("--stage1_epochs", type=int, default=2,
                   help="Number of epochs for stage1")
    p.add_argument("--stage1_lr", type=float, default=5e-4,
                   help="Learning rate for stage1")

    # Stage2 параметры
    p.add_argument("--stage2_full", dest="stage2_full", action="store_true",
                   help="Enable stage2: full fine-tuning (unfreeze base)")
    p.add_argument("--no-stage2_full", dest="stage2_full", action="store_false",
                   help="Disable stage2 training")

    p.add_argument("--stage2_epochs", type=int, default=3,
                   help="Number of epochs for stage2")
    p.add_argument("--stage2_lr", type=float, default=5e-5,
                   help="Learning rate for stage2")

    # Параметры обучения
    p.add_argument("--train_bs", type=int, default=16,
                   help="Training batch size")
    p.add_argument("--eval_bs", type=int, default=16,
                   help="Evaluation batch size")
    p.add_argument("--fp16", action="store_true",
                   help="Use FP16 mixed precision training")

    # Дополнительные параметры
    p.add_argument("--thr_search_iter", type=int, default=50,
                   help="Number of iterations for threshold search")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--run_name", type=str, default=None,
                   help="Name for the run directory (autogenerated if not provided)")
    p.add_argument("--save_parts", action="store_true",
                   help="Save model parts separately")
    p.add_argument("--load_parts_dir", type=str, default=None,
                   help="Directory to load model parts from")
    p.add_argument("--parts_to_load", type=str, default="bert,classifier,crf",
                   help="Comma-separated list of parts to load (bert,classifier,crf,other)")

    # Установка значений по умолчанию
    p.set_defaults(
        use_crf=True,
        stage1_head_only=True,
        stage2_full=True,
        optuna_pruner=True
    )

    args = p.parse_args()
    main(args)