"""Standalone loader for the character-level XSS detector.

Needs only torch, numpy and safetensors (plus huggingface_hub to download):

    from xss_detector import XSSDetector
    detector = XSSDetector.from_pretrained("<repo id or local folder>")
    detector.predict(["<img src=x onerror=alert(1)>", "<b>hello</b>"])

The weights are a safetensors file, so loading them runs no code from the download.
"""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

PAD, OTHER, VOCAB = 0, 257, 258  # 1..256 are code points 0..255; OTHER is every later code point
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str, rounds: int = 3) -> str:
    """Undo URL and HTML encoding (up to `rounds` times), drop NUL bytes, squeeze whitespace, lowercase."""
    text = str(text)
    for _ in range(rounds):
        decoded = html.unescape(unquote(text))
        if decoded == text:
            break
        text = decoded
    text = text.replace("\x00", "")
    return _WHITESPACE.sub(" ", text).strip().lower()


def windows(text: str, window: int, stride: int) -> list[str]:
    """Overlapping windows of a normalised text. Each is scored; the highest score counts."""
    if len(text) <= window:
        return [text]
    return [text[i : i + window] for i in range(0, len(text) - window + stride, stride)]


def encode(text: str, max_length: int) -> np.ndarray:
    points = np.frombuffer((text[:max_length] or " ").encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
    return np.where(points < 256, points + 1, OTHER).astype(np.int64)


class XSSNet(nn.Module):
    """Character embeddings -> convolutions over 1/3/5/7 characters -> 2 stride-2 convolutions -> Transformer encoder -> pooled head."""

    def __init__(self, max_length: int, embed_dim: int, width: int, layers: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.max_length = max_length
        self.embed = nn.Embedding(VOCAB, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, width // 4, k, padding=k // 2) for k in (1, 3, 5, 7))
        self.down = nn.ModuleList(nn.Conv1d(width, width, 3, stride=2, padding=1) for _ in range(2))
        self.position = nn.Embedding((max_length + 3) // 4, width)
        layer = nn.TransformerEncoderLayer(width, heads, 3 * width, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Logit per row; positive means XSS."""
        mask = tokens != PAD
        x = self.embed(tokens).transpose(1, 2)
        x = torch.cat([F.gelu(conv(x)) for conv in self.convs], dim=1) * mask.unsqueeze(1)
        for down in self.down:
            mask = mask[:, ::2]
            x = F.gelu(down(x)) * mask.unsqueeze(1)
        x = x.transpose(1, 2)
        x = x + self.position(torch.arange(x.size(1), device=x.device))
        x = self.norm(self.encoder(x, src_key_padding_mask=~mask))
        keep = mask.unsqueeze(-1)
        highest = x.masked_fill(~keep, -1e4).amax(dim=1)
        average = (x * keep).sum(dim=1) / keep.sum(dim=1)
        return self.head(torch.cat([highest, average], dim=-1)).squeeze(-1)


class XSSDetector:
    def __init__(self, folder: str | Path, device: str | None = None, batch_size: int = 256):
        folder = Path(folder)
        self.config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if self.config.get("architecture") != "xss_char_cnn_transformer" or self.config.get("tokenizer") != "codepoints_v1":
            raise ValueError("config.json does not describe this detector")
        net = self.config["network"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = XSSNet(net["max_length"], net["embed_dim"], net["width"], net["layers"], net["heads"])
        self.model.load_state_dict(load_file(str(folder / "model.safetensors")))
        self.model.to(self.device).eval()
        self.threshold = float(self.config["threshold"])
        self.temperature = float(self.config["temperature"])
        if not (math.isfinite(self.temperature) and self.temperature > 0 and 0 <= self.threshold <= 1):
            raise ValueError("invalid threshold or temperature in config.json")
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(cls, repo_or_folder: str, revision: str | None = None, device: str | None = None) -> "XSSDetector":
        """A local folder, or a Hugging Face repo id (downloads only config.json and model.safetensors)."""
        if Path(repo_or_folder).is_dir():
            return cls(repo_or_folder, device)
        from huggingface_hub import snapshot_download

        folder = snapshot_download(repo_or_folder, revision=revision, allow_patterns=["config.json", "model.safetensors"])
        return cls(folder, device)

    @torch.inference_mode()
    def _logits(self, chunks: list[str]) -> np.ndarray:
        rows = [encode(chunk, self.model.max_length) for chunk in chunks]
        order = np.argsort([len(row) for row in rows], kind="stable")
        out = np.zeros(len(rows))
        start = 0
        while start < len(order):
            longest = len(rows[order[min(start + self.batch_size, len(order)) - 1]])
            picks = order[start : start + max(8, min(self.batch_size, self.batch_size * 250 // longest))]
            tokens = np.zeros((len(picks), max(len(rows[i]) for i in picks)), dtype=np.int64)
            for j, i in enumerate(picks):
                tokens[j, : len(rows[i])] = rows[i]
            out[picks] = self.model(torch.from_numpy(tokens).to(self.device)).double().cpu().numpy()
            start += len(picks)
        return out

    def score(self, texts: list[str]) -> np.ndarray:
        """Calibrated probability of XSS for each text (the highest over its windows)."""
        chunks, owners = [], []
        for i, text in enumerate(texts):
            for chunk in windows(normalize(text, self.config["max_decode_rounds"]), self.config["window"], self.config["stride"]):
                chunks.append(chunk)
                owners.append(i)
        out = np.zeros(len(texts))
        if chunks:
            z = np.clip(self._logits(chunks) / self.temperature, -50, 50)
            np.maximum.at(out, owners, 1 / (1 + np.exp(-z)))
        return out

    def predict(self, texts: list[str]) -> list[dict]:
        """[{"text", "score", "is_xss"}] using the threshold tuned for 0.1% false alarms on validation data."""
        return [{"text": text, "score": round(float(s), 6), "is_xss": bool(s > self.threshold)} for text, s in zip(texts, self.score(texts))]


if __name__ == "__main__":
    import sys

    detector = XSSDetector.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else ".")
    for row in detector.predict(sys.argv[2:] or ["<img src=x onerror=alert(1)>", "<b>hello</b>"]):
        print(f"{'XSS   ' if row['is_xss'] else 'benign'}  {row['score']:.3f}  {row['text'][:100]}")
