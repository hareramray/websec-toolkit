"""A small causal Transformer with a byte tokenizer. No pretrained weights or tokenizer."""
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

ROOT = Path(__file__).resolve().parent.parent
CHAT_MODEL_PATH = ROOT / "models" / "chat_model.pkl"
PAD, USER, ASSISTANT, END = 256, 257, 258, 259
VOCAB_SIZE = 260


@dataclass
class ModelConfig:
    context_length: int = 256
    embed_dim: int = 96
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.context_length < 16 or self.embed_dim < 8 or self.heads < 1 or self.layers < 1:
            raise ValueError("Invalid model dimensions (context >= 16, width >= 8, heads/layers >= 1).")
        if self.embed_dim % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Width must be divisible by heads, and dropout must be in [0, 1).")


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.dropout = config.dropout
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(nn.Linear(config.embed_dim, 4 * config.embed_dim), nn.GELU(),
                                 nn.Linear(4 * config.embed_dim, config.embed_dim), nn.Dropout(config.dropout))

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                   dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(attention.transpose(1, 2).contiguous().reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class ScratchTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.embed_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embed_dim)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, VOCAB_SIZE, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        if tokens.size(1) > self.config.context_length:
            raise ValueError("Input exceeds context length.")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        return self.head(self.norm(self.blocks(x)))


def encode_messages(messages: list[dict], generation_prompt: bool = False) -> tuple[list[int], list[int]]:
    """Return byte/role tokens and targets; loss is computed on assistant text only."""
    tokens, targets = [], []
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("Each message needs role=user/assistant and nonempty string content.")
        part = [USER if role == "user" else ASSISTANT, *content.encode("utf-8")]
        if role == "assistant":
            part.append(END)
        tokens.extend(part)
        targets.extend([-100] + part[1:] if role == "assistant" else [-100] * len(part))
    if generation_prompt:
        tokens.append(ASSISTANT)
        targets.append(-100)
    return tokens, targets


def save_chat_model(model: ScratchTransformer, path: Path, metadata: dict) -> None:
    """Portable pickle: architecture, byte-tokenizer version, NumPy weights, training metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"kind": "scratch_byte_transformer", "format_version": 1,
              "config": asdict(model.config), "tokenizer": "utf8_bytes_v1",
              "state_dict": {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
              "metadata": metadata, "torch_version": str(torch.__version__)}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def load_chat_model(path: Path = CHAT_MODEL_PATH, device: str = "cpu") -> tuple[ScratchTransformer, dict]:
    """Only unpickle your own trusted files: pickle can execute arbitrary code."""
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or bundle.get("kind") != "scratch_byte_transformer" or bundle.get("format_version") != 1:
        raise ValueError("Not a supported scratch chat model pickle.")
    if bundle.get("tokenizer") != "utf8_bytes_v1":
        raise ValueError("Unsupported tokenizer.")
    model = ScratchTransformer(ModelConfig(**bundle["config"]))
    model.load_state_dict({name: torch.from_numpy(np.asarray(value)) for name, value in bundle["state_dict"].items()})
    return model.to(device).eval(), bundle["metadata"]


@torch.inference_mode()
def generate(model: ScratchTransformer, messages: list[dict], max_new_tokens: int = 240,
             temperature: float = 0.7, top_k: int = 40) -> str:
    if not math.isfinite(temperature) or temperature < 0 or max_new_tokens < 1 or top_k < 1:
        raise ValueError("Use finite temperature >= 0, max_new_tokens >= 1 and top_k >= 1.")
    model.eval()
    tokens, _ = encode_messages(messages, generation_prompt=True)
    device = next(model.parameters()).device
    output = []
    for _ in range(max_new_tokens):
        context = torch.tensor([tokens[-model.config.context_length:]], device=device, dtype=torch.long)
        logits = model(context)[0, -1].clone()
        logits[PAD:END] = -float("inf")  # never generate role markers or padding
        if temperature == 0:
            token = int(logits.argmax())
        else:
            logits /= temperature
            cutoff = torch.topk(logits, min(top_k, VOCAB_SIZE)).values[-1]
            logits[logits < cutoff] = -float("inf")
            token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
        if token == END:
            break
        tokens.append(token)
        output.append(token)
    return bytes(output).decode("utf-8", errors="replace").strip()
