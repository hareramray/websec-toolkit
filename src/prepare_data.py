"""Download the public XSS datasets and merge them into one clean training file.

Usage:  python src/prepare_data.py

Sources (the `source` column of the output)
  mereani     Mereani & Howe (2018), MIT licence. XSS attack URLs plus benign URLs,
              JavaScript and plain text.
  webattacks  shengqin/web-attacks-long on Hugging Face (no licence stated).
              XSS payloads plus normal sentences and JSON. Its SQL-injection rows
              are dropped: this model only answers "is this XSS?".
  twin_urls   Benign copies of the mereani attack URLs: same site, path and
              parameter names, with the payload swapped for a harmless value.
  synthetic   Generated everyday HTML, URLs and short strings (see synthetic.py).
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic  # noqa: E402
from xss_model import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "xss.csv"

FILES = {
    RAW / "mereani" / "Payloads.csv": "https://raw.githubusercontent.com/fmereani/Cross-Site-Scripting-XSS/master/XSSDataSets/Payloads.csv",
    RAW / "webattacks" / "train.csv": "https://huggingface.co/datasets/shengqin/web-attacks-long/resolve/main/train.csv",
    RAW / "webattacks" / "validate.csv": "https://huggingface.co/datasets/shengqin/web-attacks-long/resolve/main/validate.csv",
    RAW / "webattacks" / "test.csv": "https://huggingface.co/datasets/shengqin/web-attacks-long/resolve/main/test.csv",
}

_CLEAN_KEY = re.compile(r"^[\w\-.\[\]]{1,40}$")
_CLEAN_VALUE = re.compile(r"^[\w\-.+~,:/@ ]*$")
_ATTACK_HINT = re.compile(
    r"script|alert|onerror|onload|onmouse|onclick|onfocus|iframe|document|eval|prompt|confirm|marquee|cookie"
    r"|expression|data:|fromcharcode|xss|[<>\"'()`{}\\;]"
)


def download() -> None:
    for path, url in FILES.items():
        if path.exists():
            continue
        print(f"downloading {path.relative_to(ROOT)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, headers={"User-Agent": "dataset-download/1.0"}, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)


def load_mereani() -> pd.DataFrame:
    df = pd.read_csv(RAW / "mereani" / "Payloads.csv", encoding="latin-1")
    # The attack URLs were wrapped with literal "<br>" every ~100 characters when the
    # dataset was built. Left in, the model would learn "<br> means attack".
    text = df["Payloads"].astype(str).str.replace("<br>", "", regex=False)
    label = df["Class"].str.strip().map({"Malicious": 1, "Benign": 0})
    return pd.DataFrame({"text": text, "label": label, "source": "mereani"}).dropna()


def load_webattacks() -> pd.DataFrame:
    df = pd.concat(pd.read_csv(RAW / "webattacks" / f"{name}.csv") for name in ("train", "validate", "test"))
    df = df[df["text_label"].isin(["XSS", "normal"])]
    label = (df["text_label"] == "XSS").astype(int)
    return pd.DataFrame({"text": df["Payload"].astype(str), "label": label, "source": "webattacks"})


def benign_twin(url: str, rng: random.Random) -> tuple[str, str] | None:
    """Rebuild an attack URL without its payload.

    Returns (endpoint signature, benign URL). The model then sees the same sites on
    both sides and has to learn that the payload, not the address, is the attack.
    """
    if not url.startswith(("http://", "https://")):
        return None
    base, _, query = url.partition("?")
    hit = _ATTACK_HINT.search(base)
    if hit:  # the payload sits in the path: keep only the folders before it
        base, query = base[: hit.start()], ""
        if base.count("/") >= 3:
            base = base.rsplit("/", 1)[0] + "/"
    if len(base) < 12 or "." not in base or " " in base:
        return None

    params = []
    for part in query.split("&"):
        key, has_value, val = part.partition("=")
        if not has_value or not _CLEAN_KEY.match(key) or _ATTACK_HINT.search(key):
            continue
        if not _CLEAN_VALUE.match(val) or _ATTACK_HINT.search(val):
            val = synthetic.value(rng)
        params.append((key, val))
    twin = base + ("?" + "&".join(f"{k}={v}" for k, v in params) if params else "")
    return base + "?" + ",".join(sorted(k for k, _ in params)), twin


def twin_urls(attacks: pd.Series) -> pd.DataFrame:
    rng = random.Random(11)
    twins: dict[str, str] = {}
    for url in attacks:
        result = benign_twin(url, rng)
        if result:
            twins.setdefault(*result)  # one twin per endpoint
    return pd.DataFrame({"text": list(twins.values()), "label": 0, "source": "twin_urls"})


def main() -> None:
    download()
    df = pd.concat([load_mereani(), load_webattacks()], ignore_index=True)
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].map(normalize)
    print(f"dataset rows: {len(df)}")

    generated = pd.DataFrame({"text": synthetic.generate(), "label": 0, "source": "synthetic"})
    generated["text"] = generated["text"].map(normalize)
    attack_urls = df.loc[(df["source"] == "mereani") & (df["label"] == 1), "text"]
    df = pd.concat([df, twin_urls(attack_urls), generated], ignore_index=True)
    df = df[df["text"].str.len() > 0]

    # After normalising, differently encoded copies of one payload are identical.
    # Keep one, so the same payload cannot sit in both the training and the test set.
    conflicting = df.groupby("text")["label"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    df = df[~df["text"].isin(conflicting)].drop_duplicates("text")
    print(f"after cleaning: {len(df)} rows ({len(conflicting)} texts dropped for carrying both labels)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    counts = df.groupby(["source", "label"]).size().unstack(fill_value=0).rename(columns={0: "benign", 1: "xss"})
    print(counts.to_string())
    print(f"saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
