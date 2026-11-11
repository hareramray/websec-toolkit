"""Export public training splits to JSONL for the from-scratch language model.

Optional dependency: pip install datasets
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def oasst_pairs(rows):
    """Use reviewed English top-ranked replies; group related pairs by conversation tree."""
    english = {row["message_id"]: row for row in rows if row.get("lang") == "en" and not row.get("deleted")}
    for row in english.values():
        parent = english.get(row.get("parent_id"))
        if (row.get("role") == "assistant" and row.get("review_result") is True and row.get("rank") == 0
                and parent and parent.get("role") == "prompter" and row.get("text") and parent.get("text")):
            yield {"user": parent["text"], "assistant": row["text"],
                   "group": "oasst1:" + str(row.get("message_tree_id") or parent["message_id"])}


def wikitext_documents(rows):
    """Keep an article together, so neighboring paragraphs do not leak across splits."""
    parts, number = [], 0
    for row in rows:
        text = row["text"].strip()
        if not text:
            continue
        if re.match(r"^= [^=].* =\s*$", text) and parts:
            yield {"text": "\n".join(parts), "group": f"wikitext:{number}"}
            parts, number = [], number + 1
        parts.append(text)
    if parts:
        yield {"text": "\n".join(parts), "group": f"wikitext:{number}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["oasst1", "wikitext"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000, help="maximum pairs or documents to export")
    args = parser.parse_args()
    if args.limit < 8:
        parser.error("--limit must be at least 8")
    try:
        from datasets import load_dataset
    except ImportError:
        parser.error("Optional download dependency missing: python -m pip install datasets")
    if args.source == "oasst1":
        records = oasst_pairs(load_dataset("OpenAssistant/oasst1", split="train"))
    else:
        records = wikitext_documents(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", streaming=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= args.limit:
                break
    print(f"Saved {count} records from the public TRAIN split to {args.out}")
    print("Keep upstream validation/test splits out of training. See docs/DATASETS.md for attribution and licenses.")


if __name__ == "__main__":
    main()
