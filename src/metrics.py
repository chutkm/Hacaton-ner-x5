"""
Метрики: парсинг аннотаций и macro F1 для entity-level strict matching.
"""
from typing import List, Tuple, Dict
import ast
from collections import defaultdict
import re
from src.utils import fix_bio_spans 

Entity = Tuple[int, int, str]
TARGETS = ("TYPE", "BRAND", "VOLUME", "PERCENT")
_LABEL_RE = re.compile(r"^([BI])[-_]?(.+)$", flags=re.I)

def _normalize_label(raw: str) -> str:
    if raw is None:
        return None
    lab = str(raw).strip().upper()
    if lab == "" or lab == "O":
        return "O"
    m = _LABEL_RE.match(lab)
    if m:
        pref = m.group(1).upper()
        ent = m.group(2).upper()
        if ent in ("TYPE", "BRAND", "VOLUME", "PERCENT"):
            return f"{pref}-{ent}"
    if lab in ("TYPE", "BRAND", "VOLUME", "PERCENT"):
        return f"B-{lab}"
    return None


def parse_annotation(s: str) -> List[Entity]:
    """
    Преобразует строковую аннотацию вида "[(0,7,'B-TYPE'), ...]" в список кортежей.
    Нормализует метки (добавляет дефис если нужно) и возвращает только кортежи (start,end,label).
    """
    if isinstance(s, list):
        raw = s
    else:
        if not s or s == "[]":
            return []
        try:
            raw = ast.literal_eval(s)
        except Exception:
            return []

    out: List[Entity] = []
    for item in raw:
        try:
            st, en, lab = item
        except Exception:
            continue
        labn = _normalize_label(lab)
        if labn is None:
            continue
        out.append((int(st), int(en), labn))
    
    out = fix_bio_spans(out)
    
    return out

def to_strict_entities(chunks: List[Entity]) -> Dict[str, List[Tuple[int,int]]]:
    """
    Преобразует список токен/символьных чанков (start,end,label) в dict {TYPE: [(s,e), ...], ...}
    Поддерживает агрегацию B- + I- (учитывает небольшие расхождения).
    """
    by_type: Dict[str, List[Tuple[int,int]]] = defaultdict(list)
    chunks_sorted = sorted(chunks, key=lambda x: (x[0], x[1]))
    i = 0
    while i < len(chunks_sorted):
        st, en, lab = chunks_sorted[i]
        if lab == 'O' or '-' not in lab:
            i += 1
            continue
        prefix, ent = lab.split('-', 1)
        ent = ent.upper()
        if ent not in TARGETS:
            i += 1
            continue

        if prefix == 'B':
            cur_st, cur_en = st, en
            j = i + 1
            while j < len(chunks_sorted):
                st2, en2, lab2 = chunks_sorted[j]
                if '-' not in lab2:
                    break
                prefix2, ent2 = lab2.split('-', 1)
                ent2 = ent2.upper()

                if ent2 == ent and prefix2 == 'I' and st2 <= cur_en + 2:
                    cur_en = max(cur_en, en2)
                    j += 1
                else:
                    break
            by_type[ent].append((cur_st, cur_en))
            i = j
        else:
            _, ent = lab.split('-', 1)
            by_type[ent].append((st, en))
            i += 1
    return dict(by_type)

def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """
    Compute F1 safely.
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def macro_f1(true_chunks: List[List[Entity]], pred_chunks: List[List[Entity]]) -> float:
    """
    Macro-F1 averaged over TARGETS. Expects lists aligned by example index.
    true_chunks and pred_chunks — списки аннотаций (каждая аннотация — list[(start,end,label)]).
    """
    assert len(true_chunks) == len(pred_chunks)
    per_type = {t: {'tp':0,'fp':0,'fn':0} for t in TARGETS}
    for y_true, y_pred in zip(true_chunks, pred_chunks):
        te = to_strict_entities(y_true)
        pe = to_strict_entities(y_pred)
        for t in TARGETS:
            true_set = set(te.get(t, []))
            pred_set = set(pe.get(t, []))
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            per_type[t]['tp'] += tp
            per_type[t]['fp'] += fp
            per_type[t]['fn'] += fn
    f1s = [f1_from_counts(per_type[t]['tp'], per_type[t]['fp'], per_type[t]['fn']) for t in TARGETS]
    return sum(f1s) / len(f1s) if f1s else 0.0

