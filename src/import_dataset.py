"""Convert a labelled CSV or reviewed one-payload-per-line file to the XSS schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from xss_data import clean_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", required=True, help="provenance name, e.g. kaggle_xss")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--label-map", help='JSON mapping, e.g. {"normal":0,"XSS":1}')
    parser.add_argument("--label", type=int, choices=[0, 1], help="one label for ALL rows; required for .txt")
    parser.add_argument("--base", type=Path, help="optional existing canonical CSV to merge with")
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args()
    try:
        if args.input.suffix.lower() == ".txt":
            if args.label is None:
                parser.error(".txt input requires --label 0 or --label 1")
            df = pd.DataFrame({"text": args.input.read_text(encoding=args.encoding).splitlines(), "label": args.label})
        else:
            raw = pd.read_csv(args.input, keep_default_na=False, encoding=args.encoding)
            df = pd.DataFrame({"text": raw[args.text_column]})
            df["label"] = args.label if args.label is not None else raw[args.label_column]
        if args.label_map:
            mapping = json.loads(args.label_map)
            if not isinstance(mapping, dict):
                raise ValueError("--label-map must be a JSON object")
            df["label"] = df["label"].astype(str).map(mapping)
        df["source"] = args.source
        if args.base:
            df = pd.concat([pd.read_csv(args.base, keep_default_na=False), df], ignore_index=True)
        df, report = clean_rows(df)
        if df.empty:
            raise ValueError("No usable rows remain after cleaning.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(json.dumps(report, indent=2))
        print(f"Saved {args.out}")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
