"""Shared text handling for the XSS detector: normalisation and windowed scoring."""
from __future__ import annotations

import html
import re
from urllib.parse import unquote

import numpy as np

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
