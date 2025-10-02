# src/dataset.py
from typing import Tuple
import pandas as pd
import numpy as np
from datasets import Dataset
import ast
import re

from src.utils import LABELS, label2id  # единое место для LABELS и label2id

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
    # If label equals one of entities (TYPE, BRAND, ...)
    if lab in ("TYPE", "BRAND", "VOLUME", "PERCENT"):
        return f"B-{lab}"
    return None

from src.utils import fix_bio_spans  # Добавьте импорт

def parse_ann(s: str):
    if not s or s == "[]":
        return []
    try:
        arr = ast.literal_eval(s)
    except Exception:
        return []

    out = []
    for item in arr:
        try:
            a, b, c = item
        except Exception:
            continue
        label = _normalize_label(c)
        if label is None:
            continue
        if label == "O":
            continue
        try:
            out.append((int(a), int(b), label))
        except Exception:
            continue
    
    # Добавьте фикс префиксов
    out = fix_bio_spans(out)
    
    return out

def tokenize_and_align_labels(ex: dict, tokenizer, max_length: int = 128) -> dict:
    text = ex["sample"]
    ann = parse_ann(ex.get("annotation", ""))

    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        padding=False
    )
    offsets = enc["offset_mapping"]

    labels = [label2id["O"]] * len(offsets)

    ann = sorted(ann, key=lambda x: x[0])

    token_idx = 0
    for start_char, end_char, label in ann:
        if label not in label2id or label == "O":
            continue

        while token_idx < len(offsets) and offsets[token_idx][1] <= start_char:
            token_idx += 1

        if token_idx >= len(offsets):
            break

        if start_char <= offsets[token_idx][0] < end_char or (offsets[token_idx][0] < end_char and offsets[token_idx][1] > start_char):
            if "-" not in label:
                continue
            _, entity = label.split("-", 1)
            labels[token_idx] = label2id[f"B-{entity}"]
            token_idx += 1

            while token_idx < len(offsets) and offsets[token_idx][0] < end_char:
                if offsets[token_idx][1] > offsets[token_idx][0]:
                    labels[token_idx] = label2id[f"I-{entity}"]
                token_idx += 1

    for i, (a, b) in enumerate(offsets):
        if a == b:
            labels[i] = -100

    enc.pop("offset_mapping")
    enc["labels"] = labels
    enc["sample"] = text
    enc["annotation"] = ex.get("annotation", "")

    return enc

def load_hf_datasets(train_csv: str, val_csv: str, tokenizer, max_length: int = 128) -> Tuple[Dataset, Dataset]:
    df_train = pd.read_csv(train_csv, sep=";").dropna(subset=['sample', 'annotation']).reset_index(drop=True)
    df_val = pd.read_csv(val_csv, sep=";").dropna(subset=['sample', 'annotation']).reset_index(drop=True)

    ds_train = Dataset.from_pandas(df_train)
    ds_val = Dataset.from_pandas(df_val)

    ds_train = ds_train.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, max_length),
        batched=False,
        desc="Tokenizing train"
    )
    ds_val = ds_val.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, max_length),
        batched=False,
        desc="Tokenizing val"
    )

    return ds_train, ds_val
