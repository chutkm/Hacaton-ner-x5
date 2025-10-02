import os
import json
import torch
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoConfig
from src.model.modular import ModularTokenClassifier
from src.utils import (
    logits_to_labels_and_confs,
    fix_bio_spans,
    LABELS,
    id2label,
    fix_bio_labels
)

def load_model(model_dir: str):
    """
    Загружает модель с правильной обработкой CRF слоя и метаданными
    """
    if not os.path.exists(model_dir):
        raise ValueError(f"Директория модели {model_dir} не существует")
    
    meta_path = os.path.join(model_dir, "meta.json")
    use_crf = False
    threshold = 0.6
    
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        use_crf = meta.get("use_crf", False)

        bt = meta.get("best_threshold", 0.6)
        if isinstance(bt, dict):
            threshold = bt.get("thr", 0.6)
        else:
            threshold = float(bt)
    else:
        print("Предупреждение: Файл meta.json не найден. Будет использован порог по умолчанию 0.6")

    tok_dir = model_dir
    if os.path.isdir(os.path.join(model_dir, "tokenizer")):
        tok_dir = os.path.join(model_dir, "tokenizer")
    

    tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        config = AutoConfig.from_pretrained(model_dir)
    else:
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.num_labels = len(LABELS)
        config.id2label = {i: label for i, label in enumerate(LABELS)}
        config.label2id = {label: i for i, label in enumerate(LABELS)}

    try:
        model = ModularTokenClassifier.from_pretrained(
            model_dir,
            config=config,
            use_crf=use_crf,
            num_labels=len(LABELS)
        )
    except Exception:
        try:
            model = ModularTokenClassifier.from_pretrained_base(
                pretrained_path=model_dir,
                use_crf=use_crf,
                num_labels=len(LABELS)
            )
        except Exception:
            model = ModularTokenClassifier(
                config=config,
                use_crf=use_crf,
                num_labels=len(LABELS)
            )
    model.eval()
    
    return tokenizer, model, threshold

def correct_empty_spans(spans: List[Tuple[int, int, str]], sample: str) -> List[Tuple[int, int, str]]:
    """
    Корректирует пустой список спанов, возвращая [(0, len(sample), 'O')] для непустого текста.
    
    Args:
        spans: Список кортежей (start, end, label).
        sample: Входной текст (запрос).
        
    Returns:
        Список спанов, где для пустого списка и непустого текста возвращается [(0, len(sample), 'O')].
    """
    if not spans and sample and sample.strip():
        return [(0, len(sample), 'O')]
    return spans

@torch.inference_mode()
def predict_spans(
    text: str,
    model: ModularTokenClassifier,
    tokenizer,
    device: str = "cuda",
    threshold: float = 0.6,
    max_length: int = 128
) -> List[Tuple[int, int, str]]:
    """
    Предсказывает спаны для текста с использованием модели
    
    Args:
        text: Входной текст для обработки
        model: Загруженная NER-модель
        tokenizer: Токенизатор, соответствующий модели
        device: Устройство для вычислений (cpu/cuda)
        threshold: Порог уверенности для фильтрации спанов
        max_length: Максимальная длина последовательности
        
    Returns:
        Список кортежей (start, end, label) - выделенные сущности
    """

    if not text or not text.strip():
        return []
    
    inputs = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    
    offsets = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "logits"):
        logits = outputs.logits[0]
    elif "logits" in outputs:
        logits = outputs["logits"][0]
    else:
        raise ValueError("Модель не возвращает логиты")
    
    labels, confs = logits_to_labels_and_confs(logits, {i: l for i, l in enumerate(LABELS)})

    spans = []
    current_span = None
    
    for i, (a, b) in enumerate(offsets):
        if a == b or i >= len(labels) or i >= len(confs):
            continue
        
        label = labels[i]
        conf = confs[i]

        if label == "O":
            if current_span:
                spans.append(current_span)
                current_span = None
            continue
        
        if "-" not in label:
            continue
        
        prefix, entity = label.split("-", 1)

        if prefix == "B" or current_span is None or current_span["entity"] != entity:
            if current_span:
                spans.append(current_span)
            current_span = {
                "start_index": a,
                "end_index": b,
                "entity": entity,
                "original_label": label,
                "confs": [conf]
            }

        elif prefix == "I" and current_span and current_span["entity"] == entity:
            current_span["end_index"] = b
            current_span["confs"].append(conf)
    

    if current_span:
        spans.append(current_span)
    

    formatted_spans = []
    for span in spans:
        
        formatted_spans.append((span["start_index"], span["end_index"], span["original_label"]))
    
    formatted_spans = fix_bio_spans(formatted_spans, text)
    
    filtered_spans = []
    for span, orig_span in zip(formatted_spans, spans):
        avg_conf = sum(orig_span["confs"]) / len(orig_span["confs"])
        if avg_conf >= threshold:
            filtered_spans.append(span)

    filtered_spans = correct_empty_spans(filtered_spans, text)
    
    return filtered_spans