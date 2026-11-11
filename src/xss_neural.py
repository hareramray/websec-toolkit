"""Neural XSS detector: character-level convolutions feeding a Transformer encoder, trained from random weights.

The linear model in xss_model.py adds up independent character n-grams. This network
reads the same normalised characters but can combine them: which handler sits in
which tag, what the handler's code does, where a quote closes an attribute.

`NeuralXSS` offers `predict_proba`, so `xss_model.score()` and everything built on
it (window scoring, chat, the agent) works with either model.
"""
from __future__ import annotations

import math
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from xss_model import ROOT, WINDOW, hot_spans, normalize, windows

NEURAL_PATH = ROOT / "models" / "xss_neural.pkl"
KIND = "xss_char_cnn_transformer"
PAD, OTHER, VOCAB = 0, 257, 258  # 1..256 are code points 0..255; OTHER is every later code point


@dataclass
class NeuralConfig:
    max_length: int = WINDOW  # characters, the same window the linear model scores
    embed_dim: int = 128
    width: int = 192
    layers: int = 4
    heads: int = 8
    dropout: float = 0.1

    def __post_init__(self):
        if self.max_length < 8 or self.embed_dim < 8 or self.layers < 1 or self.heads < 1:
            raise ValueError("Invalid model dimensions.")
        if self.width % 4 or self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Width must be divisible by 4 and by heads, and dropout must be in [0, 1).")


def encode(texts: list[str], max_length: int) -> list[np.ndarray]:
    """One id per character. Text must already be normalised."""
    out = []
    for text in texts:
        points = np.frombuffer((text[:max_length] or " ").encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
        out.append(np.where(points < 256, points + 1, OTHER).astype(np.int64))
    return out


def pad(rows: list[np.ndarray], device) -> torch.Tensor:
    tokens = np.zeros((len(rows), max(len(row) for row in rows)), dtype=np.int64)
    for i, row in enumerate(rows):
        tokens[i, : len(row)] = row
    return torch.from_numpy(tokens).to(device)


class XSSNet(nn.Module):
    def __init__(self, config: NeuralConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(VOCAB, config.embed_dim, padding_idx=PAD)
        # Parallel convolutions read 1, 3, 5 and 7 characters at once: learned n-grams.
        self.convs = nn.ModuleList(nn.Conv1d(config.embed_dim, config.width // 4, k, padding=k // 2) for k in (1, 3, 5, 7))
        # Two stride-2 convolutions: attention then works on one position per 4 characters, 16x cheaper.
        self.down = nn.ModuleList(nn.Conv1d(config.width, config.width, 3, stride=2, padding=1) for _ in range(2))
        self.position = nn.Embedding((config.max_length + 3) // 4, config.width)
        layer = nn.TransformerEncoderLayer(config.width, config.heads, 3 * config.width, config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Sequential(nn.Linear(2 * config.width, config.width), nn.GELU(),
                                  nn.Dropout(config.dropout), nn.Linear(config.width, 1))

    def features(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = embedded.transpose(1, 2)
        x = torch.cat([F.gelu(conv(x)) for conv in self.convs], dim=1) * mask.unsqueeze(1)
        for down in self.down:  # padding is zeroed after every step, so it cannot leak into real positions
            mask = mask[:, ::2]
            x = F.gelu(down(x)) * mask.unsqueeze(1)
        x = x.transpose(1, 2)
        x = x + self.position(torch.arange(x.size(1), device=x.device))
        x = self.norm(self.encoder(x, src_key_padding_mask=~mask))
        keep = mask.unsqueeze(-1)
        highest = x.masked_fill(~keep, -1e4).amax(dim=1)
        average = (x * keep).sum(dim=1) / keep.sum(dim=1)
        return self.head(torch.cat([highest, average], dim=-1)).squeeze(-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logit per row; positive means XSS."""
        if tokens.size(1) > self.config.max_length:
            raise ValueError("Input exceeds the model's window.")
        return self.features(self.embed(tokens), tokens != PAD)


class NeuralXSS:
    """A trained network behind the `predict_proba` interface that `xss_model.score()` expects."""

    def __init__(self, model: XSSNet, temperature: float = 1.0, device: str | None = None, batch_size: int = 256, fast: bool = False):
        """`fast` uses reduced precision on a GPU. Scores then differ slightly, so it is only for picking the best epoch."""
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a positive finite value")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fast = fast and str(self.device).startswith("cuda")
        self.model = model.to(self.device).eval()
        self.temperature = temperature
        self.batch_size = batch_size

    @torch.inference_mode()
    def logits(self, chunks: list[str]) -> np.ndarray:
        """Raw logits for normalised chunks no longer than the window."""
        rows = encode(chunks, self.model.config.max_length)
        order = np.argsort([len(row) for row in rows], kind="stable")  # similar lengths together: less padding
        out = np.zeros(len(rows))
        start = 0
        while start < len(order):
            # Attention memory grows with length squared, so long chunks go in smaller batches.
            longest = len(rows[order[min(start + self.batch_size, len(order)) - 1]])
            picks = order[start : start + max(8, min(self.batch_size, self.batch_size * 250 // longest))]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.fast):
                logits = self.model(pad([rows[i] for i in picks], self.device))
            out[picks] = logits.double().cpu().numpy()
            start += len(picks)
        return out

    def predict_proba(self, chunks: list[str]) -> np.ndarray:
        scaled = np.clip(self.logits(list(chunks)) / self.temperature, -50, 50)
        attack = 1 / (1 + np.exp(-scaled))
        return np.column_stack([1 - attack, attack])

    def explain(self, text: str, top: int = 3) -> list[str]:
        """The pieces of `text` whose removal lowers the XSS score the most.

        Short stretches of the highest-scoring window are blanked out one at a time.
        A stretch matters if the score drops without it.
        """
        chunks = windows(normalize(text))
        chunk = chunks[int(np.argmax(self.logits(chunks)))]
        if not chunk:
            return []
        stride = max(1, len(chunk) // 128)
        span = max(4, 3 * stride)
        starts = list(range(0, max(1, len(chunk) - span + 1), stride))
        blanked = [chunk[:s] + " " * len(chunk[s : s + span]) + chunk[s + span :] for s in starts]
        scores = self.logits([chunk, *blanked])
        heat, cover = np.zeros(len(chunk)), np.zeros(len(chunk))
        for s, drop in zip(starts, scores[0] - scores[1:]):
            heat[s : s + span] += drop
            cover[s : s + span] += 1
        return hot_spans(chunk, heat / np.maximum(cover, 1), top)


def save_neural(model: XSSNet, threshold: float, temperature: float, path: Path = NEURAL_PATH, metadata: dict | None = None) -> None:
    """Portable pickle: architecture, NumPy weights, threshold, calibration temperature, training metadata."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite value between 0 and 1")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"kind": KIND, "format_version": 1, "config": asdict(model.config), "tokenizer": "codepoints_v1",
              "state_dict": {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
              "threshold": float(threshold), "temperature": float(temperature),
              "metadata": metadata or {}, "torch_version": str(torch.__version__)}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def from_bundle(bundle: dict, device: str | None = None) -> NeuralXSS:
    """Rebuild the network from an unpickled bundle. Called by `xss_model.load_model()`."""
    if bundle.get("kind") != KIND or bundle.get("format_version") != 1 or bundle.get("tokenizer") != "codepoints_v1":
        raise ValueError("Not a supported neural XSS model pickle.")
    model = XSSNet(NeuralConfig(**bundle["config"]))
    model.load_state_dict({name: torch.from_numpy(np.asarray(value)) for name, value in bundle["state_dict"].items()})
    return NeuralXSS(model, float(bundle.get("temperature", 1.0)), device)
