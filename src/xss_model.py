"""Shared code for the XSS detector: normalisation, windowed scoring, the pickle file, explanations."""
from __future__ import annotations

import html
import math
import os
import pickle
import re
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xss_model.pkl"
METRICS_PATH = ROOT / "models" / "xss_metrics.json"

WINDOW = 1000  # characters the model looks at in one go
STRIDE = 500  # step between windows on longer inputs
MAX_DECODE_ROUNDS = 3  # attackers double- and triple-encode payloads

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Undo the encodings used to hide a payload, then lowercase.

    `%3Cscript%3E`, `&lt;script&gt;` and `<SCRIPT>` all become `<script>`, so the
    model learns the payload rather than one particular way of writing it.
    """
    text = str(text)
    for _ in range(MAX_DECODE_ROUNDS):
        decoded = html.unescape(unquote(text))
        if decoded == text:
            break
        text = decoded
    text = text.replace("\x00", "")
    return _WHITESPACE.sub(" ", text).strip().lower()


def windows(text: str) -> list[str]:
    """Split a long normalised text into overlapping windows."""
    if len(text) <= WINDOW:
        return [text]
    return [text[i : i + WINDOW] for i in range(0, len(text) - WINDOW + STRIDE, STRIDE)]


def score(pipeline, texts, normalized: bool = False) -> np.ndarray:
    """Probability of XSS for each text.

    Long inputs are scored window by window and the highest score wins, so a
    payload cannot be hidden by padding it with thousands of harmless characters.
    """
    chunks: list[str] = []
    owners: list[int] = []
    for i, text in enumerate(texts):
        for chunk in windows(text if normalized else normalize(text)):
            chunks.append(chunk)
            owners.append(i)
    out = np.zeros(len(texts))
    if chunks:
        np.maximum.at(out, owners, pipeline.predict_proba(chunks)[:, 1])
    return out


def save_model(pipeline, threshold: float, path: Path = MODEL_PATH) -> None:
    """Write the trained model as a pickle file.

    The file holds scikit-learn objects and scalar metadata, so it can be loaded from
    any script without importing this project's code first.
    """
    import sklearn

    path = Path(path)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value between 0 and 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump({"pipeline": pipeline, "threshold": threshold, "sklearn_version": sklearn.__version__}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def default_model_path() -> Path:
    """The strongest model available: the neural one if it was trained and PyTorch is installed, else the linear one."""
    import importlib.util

    neural = ROOT / "models" / "xss_neural.pkl"
    return neural if neural.exists() and importlib.util.find_spec("torch") else MODEL_PATH


def load_model(path: Path = MODEL_PATH) -> dict:
    """Read a model pickle and return at least {"pipeline", "threshold"}.

    Works for the linear model and for the neural one from train_xss_neural.py, whose
    network is rebuilt behind the same `predict_proba` interface (needs PyTorch).

    Only load pickle files you trained yourself. Unpickling runs code stored in the
    file, so a pickle from an unknown source can take over the machine.
    """
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    if isinstance(bundle, dict) and str(bundle.get("kind", "")).startswith("xss_char_"):
        from xss_neural import from_bundle

        bundle = {"pipeline": from_bundle(bundle), "threshold": bundle.get("threshold"),
                  "kind": bundle["kind"], "metadata": bundle.get("metadata", {})}
    if not isinstance(bundle, dict) or not {"pipeline", "threshold"} <= bundle.keys():
        raise ValueError("Not an XSS model pickle: expected pipeline and threshold.")
    threshold = float(bundle["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid XSS decision threshold.")
    if not hasattr(bundle["pipeline"], "predict_proba"):
        raise ValueError("XSS pipeline does not support probability scoring.")
    return bundle


def explain(pipeline, text: str, top: int = 3) -> list[str]:
    """The pieces of `text` that pushed its XSS score up the most.

    The linear model adds a fixed amount to the score for every character n-gram.
    Those amounts are spread over the characters they cover, and the stretches with
    the highest totals are returned, strongest first. The neural model brings its
    own `explain`, which measures how the score drops when a stretch is blanked out.
    """
    if hasattr(pipeline, "explain"):
        return pipeline.explain(text, top)
    tfidf, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    chunks = windows(normalize(text))
    chunk = chunks[int(np.argmax(pipeline.predict_proba(chunks)[:, 1]))]
    row = tfidf.transform([chunk])
    pull = dict(zip(row.indices.tolist(), (row.data * clf.coef_[0][row.indices]).tolist()))

    low, high = tfidf.ngram_range
    grams = [(i, n, chunk[i : i + n]) for n in range(low, high + 1) for i in range(len(chunk) - n + 1)]
    seen = Counter(gram for _, _, gram in grams)
    heat = np.zeros(len(chunk))
    for i, n, gram in grams:
        column = tfidf.vocabulary_.get(gram)
        if column in pull:
            heat[i : i + n] += pull[column] / seen[gram] / n
    return hot_spans(chunk, heat, top)


def hot_spans(chunk: str, heat: np.ndarray, top: int = 3) -> list[str]:
    """Stretches of `chunk` whose per-character `heat` is near the maximum, strongest first."""
    if not len(heat) or heat.max() <= 0:
        return []

    hot = heat > 0.4 * heat.max()
    spans, start = [], None
    for i, flag in enumerate([*hot, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((float(heat[start:i].sum()), chunk[start:i].strip()))
            start = None
    out: list[str] = []
    for _, piece in sorted(spans, reverse=True):
        if len(piece) >= 2 and piece not in out:
            out.append(piece)
    return out[:top]
