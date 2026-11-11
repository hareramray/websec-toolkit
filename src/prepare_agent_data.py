"""Build labelled CSV rows for the detector behind the security agent.

Usage:
  python src/prepare_agent_data.py                 # writes data/processed/xss_agent.csv
  python src/prepare_agent_data.py --merge          # also merge with xss.csv -> xss_expanded.csv
  python src/prepare_agent_data.py --counts 8000    # scale the generated set

Why this exists. agent.py scores three shapes of text: parameter/field values,
request targets from access logs, and whole log lines. The first neural detector
was strong on attacks but raised false alarms on ordinary text in those shapes
(code, security talk, other-language text). This script writes rows that teach
those shapes, all in the canonical schema `text,label,source`:

  benign (label 0)  parameter values, request targets, log lines, other-language
                    text, and NON-XSS attacks (SQLi, traversal, command injection,
                    SSRF, scanner probes) -- real attacks, but not XSS
  attack (label 1)  EXISTING payloads only -- taken from the Mereani dataset and
                    from xss_augment.synthetic_attack -- wrapped into the agent's
                    shapes (as a value, a request target, or a log line)

No new payload is invented here; agent_data only places payloads the project
already has into the shapes the agent meets them in. Benign rows that happen to
contain script-capable markup are dropped (agent_data.injected), and any benign
row whose normalised text equals a known attack is dropped too, so the labels stay
clean before any train/test split.

Keep this out of the evaluation-only benchmark (prepare_external.py) and the
hand-written challenge set: those must stay independent of what the model trains on.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_data  # noqa: E402
from xss_augment import synthetic_attack  # noqa: E402
from xss_data import clean_rows  # noqa: E402
from xss_model import ROOT, normalize  # noqa: E402

DATA_PATH = ROOT / "data" / "processed" / "xss.csv"
OUT = ROOT / "data" / "processed" / "xss_agent.csv"
EXPANDED = ROOT / "data" / "processed" / "xss_expanded.csv"
SEED = 101


def dataset_attacks(base: Path) -> list[str]:
    """Attack payloads already in the project's processed data (label 1)."""
    if not base.exists():
        return []
    df = pd.read_csv(base, keep_default_na=False)
    return df.loc[pd.to_numeric(df["label"], errors="coerce") == 1, "text"].astype(str).tolist()


def build(counts: int, base: Path, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    # Benign rows, in the agent's shapes. `other_attack` is non-XSS attack traffic:
    # it is labelled 0 because this detector's job is XSS only.
    benign = {
        "agent_value": (agent_data.benign_value, counts),
        "agent_request": (agent_data.benign_request, counts),
        "agent_line": (agent_data.benign_line, int(counts * 0.8)),
        "agent_multilingual": (agent_data.multilingual, int(counts * 0.5)),
        "agent_other_attack": (agent_data.other_attack, int(counts * 0.6)),
    }
    for source, (generator, number) in benign.items():
        for _ in range(number):
            text = generator(rng)
            if not agent_data.injected(text):  # never let a real payload in as benign
                rows.append({"text": text, "label": 0, "source": source})

    # Attack rows: existing payloads only, placed into the agent's shapes.
    pool = dataset_attacks(base)
    rng.shuffle(pool)
    sampled = pool[: counts * 2] if pool else []
    generated = [synthetic_attack(rng) for _ in range(counts)]
    payloads = sampled + generated
    rng.shuffle(payloads)
    for payload in payloads:
        shape = rng.random()
        if shape < 0.5:
            text, source = agent_data.as_value(payload, rng), "agent_attack_value"
        elif shape < 0.8:
            text, source = agent_data.as_request(payload, rng), "agent_attack_request"
        else:
            text, source = agent_data.as_line(payload, rng), "agent_attack_line"
        rows.append({"text": text, "label": 1, "source": source})

    df = pd.DataFrame(rows)

    # Drop any benign row whose normalised text matches a known attack payload.
    attack_norm = {normalize(p) for p in payloads}
    df["_norm"] = df["text"].map(normalize)
    conflict = (df["label"] == 0) & df["_norm"].isin(attack_norm)
    if conflict.any():
        print(f"dropped {int(conflict.sum())} benign row(s) colliding with a known attack")
    df = df[~conflict].drop(columns="_norm")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--counts", type=int, default=6000, help="base count for the benign value/request generators")
    parser.add_argument("--base", type=Path, default=DATA_PATH, help="processed XSS CSV to draw real attack payloads from")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--merge", action="store_true", help="also concatenate with --base and write a cleaned xss_expanded.csv")
    parser.add_argument("--expanded", type=Path, default=EXPANDED)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.counts < 1:
        parser.error("--counts must be positive")

    raw = build(args.counts, args.base, args.seed)
    cleaned, report = clean_rows(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.out, index=False, encoding="utf-8")
    counts = cleaned.groupby(["source", "label"]).size().unstack(fill_value=0).rename(columns={0: "benign", 1: "xss"})
    print(counts.to_string())
    print(f"\ncleaning: {report}")
    print(f"saved {args.out.relative_to(ROOT)} ({len(cleaned)} rows)")

    if args.merge:
        if not args.base.exists():
            parser.error(f"--merge needs {args.base}; run src/prepare_data.py first")
        merged = pd.concat([pd.read_csv(args.base, keep_default_na=False), cleaned], ignore_index=True)
        merged, mreport = clean_rows(merged)
        merged.to_csv(args.expanded, index=False, encoding="utf-8")
        print(f"\nmerged with {args.base.name}: {mreport}")
        print(f"saved {args.expanded.relative_to(ROOT)} ({len(merged)} rows)")
        print(f"\nTrain on it, for example:\n"
              f"  .\\.venv\\Scripts\\python.exe src\\train_xss.py --data {args.expanded.relative_to(ROOT)} --out models\\xss_expanded.pkl\n"
              f"  .\\.venv\\Scripts\\python.exe src\\train_xss_neural.py --data {args.expanded.relative_to(ROOT)} --out models\\xss_neural_expanded.pkl --baseline models\\xss_expanded.pkl")


if __name__ == "__main__":
    main()
