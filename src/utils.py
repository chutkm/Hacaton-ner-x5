# src/utils.py
from typing import List, Tuple, Dict, Optional
import os
import json
from datetime import datetime
import torch

# Единое определение меток для всего проекта
LABELS: List[str] = [
    "O",
    "B-TYPE", "I-TYPE",
    "B-BRAND", "I-BRAND",
    "B-VOLUME", "I-VOLUME",
    "B-PERCENT", "I-PERCENT"
    
]
id2label: Dict[int, str] = {i: l for i, l in enumerate(LABELS)}
label2id: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}

def compute_class_weights_from_counts(counts: Dict[str, int]) -> torch.Tensor:
    """
    Compute class weights as total / (num_classes * count_c)
    counts: dict mapping label string (from LABELS) to count (int)
    returns tensor of shape (num_labels,) in the same order as LABELS
    """
    total = sum(counts.get(l, 0) for l in LABELS)
    num_classes = len(LABELS)
    # Avoid division by zero
    weights = []
    for l in LABELS:
        cnt = counts.get(l, 0)
        if cnt <= 0:
            w = float(total)  # very large (fallback)
        else:
            w = float(total) / (num_classes * float(cnt))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float)

def fix_bio_labels(labels: List[str]) -> List[str]:
    """
    Исправляет последовательность BIO-меток:
    - I-entity без предшествующего B-entity -> превращаем в B-entity
    - подряд идущие B-entity того же типа (внутри одной сущности) -> второй превращаем в I-
    - I-entity после O или другого типа -> превращаем в B-
    Работает со списком строк меток (например, ["O","B-TYPE","I-TYPE",...])
    """
    fixed: List[str] = []
    prev = "O"
    for lab in labels:
        if lab is None:
            lab = "O"
        if lab == "O" or "-" not in lab:
            fixed.append("O")
            prev = "O"
            continue
        prefix, ent = lab.split("-", 1)
        prefix = prefix.upper()
        ent = ent.upper()
        if prefix == "B":
            # если предыдущая метка — такой же B того же типа, возможно это продолжение -> превратить в I
            if prev != "O" and "-" in prev:
                p_pref, p_ent = prev.split("-", 1)
                if p_ent == ent and p_pref == "B":
                    # считаем, что модель поставила B вместо I — исправляем
                    fixed.append(f"I-{ent}")
                    prev = f"I-{ent}"
                    continue
            fixed.append(f"B-{ent}")
            prev = f"B-{ent}"
        elif prefix == "I":
            # если предыдущая не I того же типа -> превратить в B
            if prev == "O" or "-" not in prev:
                fixed.append(f"B-{ent}")
                prev = f"B-{ent}"
            else:
                p_pref, p_ent = prev.split("-", 1)
                if p_ent != ent:
                    fixed.append(f"B-{ent}")
                    prev = f"B-{ent}"
                else:
                    # prev has same ent
                    if p_pref == "B" or p_pref == "I":
                        fixed.append(f"I-{ent}")
                        prev = f"I-{ent}"
                    else:
                        fixed.append(f"B-{ent}")
                        prev = f"B-{ent}"
        else:
            fixed.append("O")
            prev = "O"
    return fixed

def logits_to_labels_and_confs(logits: torch.Tensor, id2label_map: Dict[int,str]) -> Tuple[List[str], List[float]]:
    """
    logits: тензор shape (seq_len, num_labels) или (num_labels,) для одного токена.
    Возвращает: (labels_list, confs_list) длины seq_len.
    confs — вероятности выбранного класса (float).
    """
    # если одномерный — привести к двумерному для унификации
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = torch.softmax(logits, dim=-1).cpu()
    confs, ids = probs.max(dim=-1)
    labels = [id2label_map[int(i.item())] for i in ids]
    return labels, confs.tolist()

def tokens_to_spans(offsets: List[Tuple[int,int]], labels: List[str]) -> List[Tuple[int,int,str]]:
    """
    Конвертирует последовательность токенов с метками в список span'ов в виде
    (start_char, end_char, original_label).
    original_label — метка B-ENTITY (берётся из первого токена сущности).
    Пропускает метки 'O' и спецтокены (offsets с a==b).
    """
    spans: List[Tuple[int,int,str]] = []
    current = None  # (start, end, original_label)
    for (start, end), lab in zip(offsets, labels):
        # пропускаем спецтокены и O
        if start == end or lab == "O" or "-" not in lab:
            if current is not None:
                spans.append(current)
                current = None
            continue

        prefix, ent = lab.split("-", 1)
        if prefix == "B" or current is None or (current is not None and not current[2].endswith(ent)):
            # начинаем новую сущность; сохраняем B-метку (оригинальный префикс из первого токена)
            if current is not None:
                spans.append(current)
            current = (start, end, f"B-{ent}")
        else:
            # продолжаем существующую сущность (I-)
            current = (current[0], end, current[2])

    if current is not None:
        spans.append(current)

    return spans

# Utilities for saving/loading model parts and run dir (как было у тебя)
def _split_state_dict_by_parts(state: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    bert_parts = {}
    classifier_parts = {}
    crf_parts = {}
    other_parts = {}
    for k,v in state.items():
        kl = k.lower()
        if k.startswith("classifier.") or ".classifier." in k:
            classifier_parts[k] = v
        elif "crf" in kl or "transitions" in kl:
            crf_parts[k] = v
        elif any(x in kl for x in ("embeddings", "encoder", "layer.", "pooler", "roberta.", "bert.", "xlm_roberta", "base_model", "embedding")):
            bert_parts[k] = v
        else:
            other_parts[k] = v
    return {"bert": bert_parts, "classifier": classifier_parts, "crf": crf_parts, "other": other_parts}

def save_model_parts(model: torch.nn.Module, out_dir: str, save_full: bool = True, tokenizer=None, meta: Optional[dict] = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()
    if save_full:
        torch.save(sd, os.path.join(out_dir, "full_model.pth"))
    parts = _split_state_dict_by_parts(sd)
    for name, part_sd in parts.items():
        if part_sd:
            torch.save(part_sd, os.path.join(out_dir, f"{name}.pth"))
    if meta is not None:
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(out_dir, "tokenizer"))

def load_model_parts(model: torch.nn.Module, parts_dir: str, parts_to_load: Optional[List[str]] = None, device: str = "cpu") -> List[str]:
    if parts_to_load is None or "all" in parts_to_load:
        parts_to_load = ["bert", "classifier", "crf", "other"]
    loaded = []
    base_sd = model.state_dict()
    full_path = os.path.join(parts_dir, "full_model.pth")
    if os.path.exists(full_path) and ("full" in parts_to_load or not any(os.path.exists(os.path.join(parts_dir, p + ".pth")) for p in ["bert","classifier","crf","other"])):
        full_sd = torch.load(full_path, map_location=device)
        model.load_state_dict(full_sd, strict=False)
        loaded = ["full"]
        return loaded

    for part in parts_to_load:
        path = os.path.join(parts_dir, f"{part}.pth")
        if not os.path.exists(path):
            continue
        part_sd = torch.load(path, map_location=device)
        for k, v in part_sd.items():
            if k in base_sd:
                base_sd[k] = v.to(device)
            else:
                base_sd[k] = v.to(device)
        loaded.append(part)
    model.load_state_dict(base_sd, strict=False)
    return loaded

def make_run_dir(base_dir: str = "runs", run_name: Optional[str] = None) -> str:
    os.makedirs(base_dir, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

    
def fix_bio_spans(spans: List[Tuple[int, int, str]], text: Optional[str] = None) -> List[Tuple[int, int, str]]:
    """
    Фиксит префиксы B/I в списке char-спанов (Entity).
    - Сортирует по start.
    - Если consecutive спаны adjacent (start <= prev_end +1) и same entity: первый B-, остальные I-.
    - I- без prev -> B-.
    - Если spans пустой и передан text, возвращает [(0, len(text), 'O')].
    - Возвращает fixed list спанов с обновлёнными labels.
    """
    if not spans and text:
        return [(0, len(text), 'O')]
    
    if not spans:
        return []
    
    sorted_spans = sorted(spans, key=lambda x: x[0])
    
    fixed: List[Tuple[int, int, str]] = []
    prev_ent: Optional[str] = None
    
    for s, e, l in sorted_spans:
        if l == "O" or "-" not in l:
            fixed.append((s, e, "O"))
            prev_ent = None
            continue
        
        pref, ent = l.split("-", 1)
        ent = ent.upper()
        
        is_continuation = (prev_ent == ent) and fixed and (s <= fixed[-1][1] + 1)
        
        if is_continuation:
            new_l = f"I-{ent}"
        else:
            new_l = f"B-{ent}"
        
        fixed.append((s, e, new_l))
        prev_ent = ent
    
    return fixed