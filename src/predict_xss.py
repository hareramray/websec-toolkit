"""Score strings with the trained XSS detector.

Usage:
  python src/predict_xss.py "<script>alert(1)</script>" "hello world"
  python src/predict_xss.py --file inputs.txt        (one input per line)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xss_model import default_model_path, load_model, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="strings to score")
    parser.add_argument("--file", type=Path, help="text file with one input per line")
    parser.add_argument("--model", type=Path, help="default: the neural model if trained, else the linear one")
    args = parser.parse_args()
    args.model = args.model or default_model_path()

    inputs = list(args.inputs)
    if args.file:
        inputs += [line for line in args.file.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not inputs:
        parser.error("give at least one string, or --file")
    if not args.model.exists():
        sys.exit(f"no model at {args.model}: run  python src/train_xss.py  first")

    bundle = load_model(args.model)
    scores = score(bundle["pipeline"], inputs)
    for text, s in zip(inputs, scores):
        verdict = "XSS   " if s > bundle["threshold"] else "benign"
        shown = text if len(text) <= 100 else text[:97] + "..."
        print(f"{verdict}  {s:.3f}  {shown}")


if __name__ == "__main__":
    main()
