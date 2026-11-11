"""Package the neural XSS detector for the Hugging Face Hub, and optionally upload it.

Usage:
  python src/export_hf.py --repo-id your-name/xss-detector              (build dist/hf only)
  python src/export_hf.py --repo-id your-name/xss-detector --push       (build, then upload; private)
  python src/export_hf.py --repo-id your-name/xss-detector --push --public

The folder holds model.safetensors (weights, no pickle), config.json (architecture,
threshold, calibration, input handling), xss_detector.py (a loader that needs only
torch, numpy and safetensors), metrics.json and README.md (the model card, built
from the saved metrics). Uploading needs `hf auth login` first.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xss_model import MAX_DECODE_ROUNDS, ROOT, STRIDE, WINDOW  # noqa: E402
from xss_neural import KIND, NEURAL_PATH  # noqa: E402

OUT = ROOT / "dist" / "hf"
LOADER = ROOT / "huggingface" / "xss_detector.py"
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$")


def public_metrics(metrics: dict) -> dict:
    """The metrics that are safe to publish: no local file paths, no training history."""
    keep = {"model", "threshold", "target_fpr", "rows", "sources", "cleaning", "held_out_test", "challenge_set",
            "external_benchmark", "linear_baseline"}
    out = {key: value for key, value in metrics.items() if key in keep}
    if isinstance(out.get("linear_baseline"), dict):
        out["linear_baseline"] = {key: value for key, value in out["linear_baseline"].items() if key != "path"}
    text = json.dumps(out)
    if re.search(r"[A-Za-z]:\\\\|/home/|/Users/", text):
        raise ValueError("metrics still contain a local file path; refusing to publish them")
    return out


def pct(part: float, whole: float, digits: int = 1) -> str:
    return f"{100 * part / whole:.{digits}f}%" if whole else "n/a"


def model_card(repo_id: str, license_id: str, config: dict, m: dict) -> str:
    net, info = config["network"], m["model"]
    held, ext, chal = m["held_out_test"], m["external_benchmark"], m["challenge_set"]
    base = m.get("linear_baseline") or {}
    bh, be, bc = base.get("held_out_test", {}), base.get("external_benchmark", {}), base.get("challenge_set", {})
    hc, bhc = held["confusion"], bh.get("confusion", {})
    attacks, benign = hc["tp"] + hc["fn"], hc["fp"] + hc["tn"]

    def row(name: str, neural: str, linear: str) -> str:
        return f"| {name} | {neural} | {linear} |"

    rows = [
        row("Held-out attacks caught", f"{hc['tp']:,} / {attacks:,} ({pct(hc['tp'], attacks)})", f"{bhc.get('tp', 0):,} ({pct(bhc.get('tp', 0), attacks)})" if bhc else "n/a"),
        row("Held-out false alarms", f"{hc['fp']} / {benign:,} ({pct(hc['fp'], benign, 2)})", f"{bhc.get('fp', 0)} ({pct(bhc.get('fp', 0), benign, 2)})" if bhc else "n/a"),
        row("Held-out detection at exactly 0.1% false alarms", f"{100 * held['detection_at_0.1pct_false_alarms']:.1f}%",
            f"{100 * bh['detection_at_0.1pct_false_alarms']:.1f}%" if bh else "n/a"),
        row("External attacks caught (SecLists)", f"{ext['caught']:,} / {ext['attacks']:,} ({pct(ext['caught'], ext['attacks'])})",
            f"{be['caught']:,} ({pct(be['caught'], ext['attacks'])})" if be else "n/a"),
        row("External attacks matching no training-pool row", f"{ext['unseen_caught']:,} / {ext['unseen_attacks']:,} ({pct(ext['unseen_caught'], ext['unseen_attacks'])})",
            f"{be['unseen_caught']:,} ({pct(be['unseen_caught'], ext['unseen_attacks'])})" if be else "n/a"),
        row("External benign false alarms (WikiText, OpenAssistant)", f"{ext['false_alarms']} / {ext['benign']:,}", f"{be['false_alarms']}" if be else "n/a"),
        row("Hand-written challenge attacks caught", f"{chal['caught']} / {chal['attacks']}", f"{bc['caught']}" if bc else "n/a"),
        row("Hand-written challenge false alarms", f"{len(chal['false_alarms'])} / {chal['benign']}", f"{len(bc['false_alarms'])}" if bc else "n/a"),
    ]
    missed = "".join(f"\n  - `{text}`" for text in chal["missed"]) or " none"

    sources = set(m.get("sources", []))
    known_sources = {
        "mereani": "| [Mereani & Howe XSS dataset](https://github.com/fmereani/Cross-Site-Scripting-XSS) | XSS attack URLs and payloads; benign URLs, JavaScript and text | MIT |",
        "synthetic": "| Generated benign samples | Everyday HTML, URLs, code and prose that a naive model mistakes for attacks | Created for this model |",
        "twin_urls": "| Benign copies of attack URLs | Same sites and parameter names with the payload replaced | Derived from Mereani |",
        "agent_value": "| Generated parameter and form-field values | Ordinary values the agent scores, so the model stops flagging them | Created for this model |",
        "agent_request": "| Generated request targets | Ordinary request targets as access logs record them | Created for this model |",
        "agent_line": "| Generated log lines | Application, server, mail and build log lines | Created for this model |",
        "agent_multilingual": "| Generated non-English text | Text in other scripts and languages that must not read as an attack | Created for this model |",
        "agent_other_attack": "| Generated non-XSS attacks (benign here) | SQLi, traversal, command injection, SSRF and scanner probes, labelled benign for an XSS-only detector | Created for this model |",
        "agent_attack_value": "| XSS payloads as field values | Existing payloads placed into the agent's value shape | Derived from Mereani (MIT) and generated templates |",
        "agent_attack_request": "| XSS payloads in request targets | Existing payloads placed into one URL parameter | Derived from Mereani (MIT) and generated templates |",
        "agent_attack_line": "| XSS payloads in log lines | Existing payloads placed into a log line | Derived from Mereani (MIT) and generated templates |",
    }
    unknown = sources - known_sources.keys()
    if unknown:  # a source added with import_dataset.py: its license must be written into the card by hand
        raise ValueError(f"no license entry for training source(s) {', '.join(sorted(unknown))}; add them to export_hf.py")
    data_rows = [known_sources[source] for source in known_sources if source in sources]
    data_note = "Every training row comes from an MIT-licensed source or was generated for this model.\n"
    by_benign = ext.get("benign_sources", {})
    chat, wiki = by_benign.get("oasst", {}), by_benign.get("wikitext", {})
    code_note = (f"\n- **Programming help text is the riskiest benign input.** {chat['false_alarms']} of {chat['benign']:,} OpenAssistant\n"
                 f"  messages were flagged (mostly answers that mix prose and code blocks), against\n"
                 f"  {wiki.get('false_alarms', 0)} of {wiki.get('benign', 0):,} Wikipedia paragraphs.") if chat.get("false_alarms") else ""

    return f"""---
license: {license_id}
pipeline_tag: text-classification
tags:
- security
- xss
- cross-site-scripting
- web-security
- character-level
- pytorch
- safetensors
---

# Character-level XSS detector

A small neural network ({info['parameters']:,} parameters) that scores how likely a
string is a cross-site scripting (XSS) payload. It reads raw characters after
undoing URL and HTML encoding, so `%3Cscript%3E`, `&lt;script&gt;` and `<ScRiPt>`
look the same to it. It was trained from random weights; no pretrained model or
tokenizer is involved.

Use it to triage inputs: request parameters, form fields, log lines, user-supplied
HTML. A high score means "looks like an injection payload". It does not prove that
a payload would execute, and a low score does not prove an input is safe. Keep
context-aware output encoding and a Content Security Policy in place regardless.

## Quick start

```bash
pip install torch numpy safetensors huggingface_hub
```

```python
from huggingface_hub import hf_hub_download
import importlib.util

path = hf_hub_download("{repo_id}", "xss_detector.py")
spec = importlib.util.spec_from_file_location("xss_detector", path)
xss_detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xss_detector)

detector = xss_detector.XSSDetector.from_pretrained("{repo_id}")
detector.predict(["<img src=x onerror=alert(1)>", "<b>hello</b>"])
# [{{'text': '<img src=x onerror=alert(1)>', 'score': ..., 'is_xss': True}},
#  {{'text': '<b>hello</b>', 'score': ..., 'is_xss': False}}]
```

`xss_detector.py` is plain Python in this repository; read it before running it.
The weights are in `model.safetensors`, so loading them executes no code.
`detector.score(texts)` returns calibrated probabilities; `predict` applies the
threshold {config['threshold']:.4f}, tuned on validation data for 0.1% false alarms.
Pick your own threshold if your traffic differs.

## How it works

1. **Normalise:** URL-decode and HTML-unescape up to {config['max_decode_rounds']} times, drop NUL bytes,
   collapse whitespace, lowercase.
2. **Window:** inputs longer than {config['window']} characters are split into windows with a
   stride of {config['stride']}; the highest window score counts, so padding a payload with
   harmless text does not hide it.
3. **Network:** one token per character (code points 0-255 each have their own token;
   every other character shares one), convolutions over 1, 3, 5 and 7 characters, two
   stride-2 convolutions, {net['layers']} Transformer encoder blocks (width {net['width']}, {net['heads']} heads), and a
   head over max- and mean-pooled features.
4. **Calibrate:** logits are divided by a temperature of {config['temperature']:.3f} fitted on
   validation data.

Trained for {info['epochs']} epochs (epoch {info['best_epoch']} kept) on an NVIDIA RTX 5050 laptop GPU in about
{round(info['training_seconds'] / 60)} minutes, with AdamW, a cosine learning-rate schedule and an averaged copy of
the weights.

## Training data

| Source | Used for | License |
|---|---|---|
{chr(10).join(data_rows)}

After normalising, exact duplicates and texts with conflicting labels were removed,
leaving {m['rows']:,} rows, split 70/15/15 into training, validation and test.
During training, each epoch adds varied copies of training attacks (real actions
such as cookie theft or redirects in place of `alert(1)`, other event handlers,
payloads placed inside ordinary text, URLs or JSON), template-built attacks, and
benign look-alikes (prose and code that mention `onclick`, `<script>` or cookies).
Site names in URLs are swapped in both classes so that no domain predicts a label.
Validation and test rows are never varied.

{data_note}
## Evaluation

Both columns use the same data and splits. The linear baseline is character TF-IDF
with logistic regression, trained on the same training rows, at its own
validation-tuned threshold.

| Measurement | This model | Linear baseline |
|---|---|---|
{chr(10).join(rows)}

**External benchmark:** {ext['attacks']:,} payloads from nine
[SecLists](https://github.com/danielmiessler/SecLists/tree/master/Fuzzing/XSS) lists (MIT) and
{ext['benign']:,} paragraphs and messages from WikiText and OpenAssistant. Nothing was trained or
tuned on it. Many SecLists payloads also occur in the training sources, so the row
counting only lines that match no row of the training pool is the stricter one.

**Challenge set:** {chal['attacks'] + chal['benign']} hand-written strings. Its past mistakes guided the training-time
variation, so treat it as a regression check, not an independent test. Missed:{missed}

## Limitations

- **Unfamiliar benign text raises false alarms.** Benign training text is mostly URLs,
  JavaScript, reviews and generated HTML. Measure the false-alarm rate on your own
  traffic and adjust the threshold before blocking anything.{code_note}
- **Injected markup counts as an attack.** `<iframe src=...>`, `<script src=...>` and
  `<form>` in an *input* are scored high, as the training data labels them. Do not
  run it over whole web pages.
- **Non-Latin text is coarse.** Every character above code point 255 is one token, so
  Cyrillic, Devanagari or Chinese text looks alike. One external false alarm was a
  Russian-language message.
- **A score is evidence, not proof.** It does not know where the input is written
  (HTML body, attribute, script, URL), which decides whether it can execute.
- **Exact deduplication only.** Related payload families can sit in both training
  and test data, so held-out numbers are optimistic. The external benchmark is the
  better estimate.

## Intended use

Defensive: triaging inputs and logs, flagging suspicious values for review, and
measuring how well a filter copes with encoded payloads. It is not a web application
firewall on its own, and it was not built or evaluated for generating payloads.
"""


def build(model_path: Path, out: Path, repo_id: str, license_id: str) -> Path:
    from safetensors.numpy import save_file

    with open(model_path, "rb") as handle:  # only your own trusted pickle
        bundle = pickle.load(handle)
    if bundle.get("kind") != KIND or bundle.get("format_version") != 1:
        raise ValueError(f"{model_path} is not a neural XSS model pickle")
    metrics_path = model_path.with_name(model_path.stem.replace("_model", "") + "_metrics.json")
    metrics = public_metrics(json.loads(metrics_path.read_text(encoding="utf-8")))

    folder = out / repo_id.split("/")[-1]
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    tensors = {name: np.ascontiguousarray(value) for name, value in bundle["state_dict"].items()}
    save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    config = {"architecture": KIND, "format_version": 1, "tokenizer": "codepoints_v1",
              "network": {key: bundle["config"][key] for key in ("max_length", "embed_dim", "width", "layers", "heads")},
              "threshold": float(bundle["threshold"]), "temperature": float(bundle["temperature"]),
              "window": WINDOW, "stride": STRIDE, "max_decode_rounds": MAX_DECODE_ROUNDS,
              "labels": {"0": "benign", "1": "xss"}, "initialization": "random_weights"}
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shutil.copyfile(LOADER, folder / "xss_detector.py")
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (folder / "README.md").write_text(model_card(repo_id, license_id, config, metrics), encoding="utf-8")
    return folder


def push(folder: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model", commit_message="Upload character-level XSS detector",
                      ignore_patterns=["__pycache__/*", "*.pyc"])  # importing xss_detector.py locally leaves these behind
    return f"https://huggingface.co/{repo_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="your-hf-name/model-name")
    parser.add_argument("--model", type=Path, default=NEURAL_PATH)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--license", default="mit", help="license id for the model card, e.g. mit or apache-2.0")
    parser.add_argument("--push", action="store_true", help="upload after building (needs `hf auth login`)")
    parser.add_argument("--public", action="store_true", help="with --push: make the repo public (default: private)")
    args = parser.parse_args()
    if not _REPO_ID.match(args.repo_id):
        parser.error("--repo-id must look like name/model-name")
    try:
        folder = build(args.model, args.out, args.repo_id, args.license)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"built {folder}:")
    for path in sorted(folder.iterdir()):
        print(f"  {path.name:<20} {path.stat().st_size:>12,} bytes")
    if args.push:
        print(f"uploaded to {push(folder, args.repo_id, private=not args.public)} ({'public' if args.public else 'private'})")


if __name__ == "__main__":
    main()
