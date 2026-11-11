"""Build an evaluation-only XSS benchmark from sources that are never trained on.

Usage:  python src/prepare_external.py

Attacks  SecLists Fuzzing/XSS payload lists (MIT licence), one payload per line.
Benign   WikiText paragraphs and OpenAssistant messages, taken from the files that
         prepare_chat_data.py exported. Skipped when those files are absent.

Writes data/processed/xss_external.csv. The trainers only read this file to report
results. Nothing in it is fitted on, and no threshold is tuned on it.

Limits: payload lists contain a few inert fragments, and many lists share payloads
with the public training sets. The trainers therefore report the lines that are not
exact normalised matches of any project row separately.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xss_model import ROOT, normalize  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "xss_external.csv"
BASE = "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/XSS/"
LISTS = {
    "brutelogic": "robot-friendly/XSS-BruteLogic.txt",
    "jhaddix": "robot-friendly/XSS-Jhaddix.txt",
    "rsnake": "robot-friendly/XSS-RSNAKE.txt",
    "somdev": "robot-friendly/XSS-Somdev.txt",
    "mario": "robot-friendly/XSS-Vectors-Mario.txt",
    "payloadbox": "robot-friendly/XSS-payloadbox.txt",
    "portswigger": "robot-friendly/XSS-Cheat-Sheet-PortSwigger.txt",
    "ofjaaah": "robot-friendly/XSS-OFJAAAH.txt",
    "polyglots": "Polyglots/XSS-Polyglots.txt",
}
MIN_LENGTH = 6  # shorter lines are fragments such as `">`, not payloads
BENIGN_PER_SOURCE = 5000


def attacks() -> pd.DataFrame:
    rows = []
    for name, path in LISTS.items():
        target = RAW / "seclists" / Path(path).name
        if not target.exists():
            print(f"downloading {target.relative_to(ROOT)}")
            target.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(BASE + path, headers={"User-Agent": "dataset-download/1.0"}, timeout=120)
            response.raise_for_status()
            target.write_bytes(response.content)
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        rows += [{"text": line, "label": 1, "source": f"seclists_{name}"} for line in lines]
    return pd.DataFrame(rows)


def benign() -> pd.DataFrame:
    rng = random.Random(23)
    rows = []
    wikitext, oasst = RAW / "wikitext_train.jsonl", RAW / "oasst_train.jsonl"
    if wikitext.exists():
        paragraphs = [line for record in map(json.loads, wikitext.read_text(encoding="utf-8").splitlines())
                      for line in record["text"].split("\n") if 40 <= len(line) <= 1500]
        rows += [{"text": text, "label": 0, "source": "wikitext"} for text in rng.sample(paragraphs, min(BENIGN_PER_SOURCE, len(paragraphs)))]
    if oasst.exists():
        messages = [record[role] for record in map(json.loads, oasst.read_text(encoding="utf-8").splitlines())
                    for role in ("user", "assistant") if isinstance(record.get(role), str)]
        rows += [{"text": text, "label": 0, "source": "oasst"} for text in rng.sample(messages, min(BENIGN_PER_SOURCE, len(messages)))]
    if not rows:
        print("no benign text found: run prepare_chat_data.py for wikitext and oasst1 to add a false-alarm check")
    return pd.DataFrame(rows, columns=["text", "label", "source"])


def main() -> None:
    df = pd.concat([attacks(), benign()], ignore_index=True)
    df["text"] = df["text"].map(normalize)
    df = df[df["text"].str.len() >= MIN_LENGTH].drop_duplicates("text")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(df.groupby(["source", "label"]).size().to_string())
    print(f"saved {OUT.relative_to(ROOT)} ({len(df)} rows)")


if __name__ == "__main__":
    main()
