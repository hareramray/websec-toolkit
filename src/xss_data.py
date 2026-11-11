"""Validate and deduplicate XSS rows before they reach any train/test split."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from xss_model import normalize


def clean_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    df = df.copy()
    if df["text"].isna().any():
        raise ValueError("Text values must not be null.")
    labels = pd.to_numeric(df["label"], errors="coerce")
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels must be 0 (benign) or 1 (XSS); map other labels before training.")
    df["label"] = labels.astype(int)
    df["text"] = df["text"].map(normalize)
    if "source" not in df:
        df["source"] = "custom"
    df["source"] = df["source"].fillna("custom").astype(str).str.strip().replace("", "custom")
    before = len(df)
    df = df[df["text"].str.len() > 0]
    empty = before - len(df)
    conflicts = df.groupby("text")["label"].nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    df = df[~df["text"].isin(conflicts)]
    duplicates = int(df.duplicated("text").sum())
    df = df.drop_duplicates("text").reset_index(drop=True)
    return df[["text", "label", "source"]], {
        "input_rows": before, "empty_rows": empty, "conflicting_texts": len(conflicts),
        "duplicate_rows": duplicates, "output_rows": len(df),
    }


def read_rows(path: Path) -> tuple[pd.DataFrame, dict]:
    return clean_rows(pd.read_csv(path, keep_default_na=False))
