"""Train a tiny text-generating Transformer from random weights, or continue your own checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from chat_model import (CHAT_MODEL_PATH, END, PAD, ROOT, ModelConfig, ScratchTransformer,
                        encode_messages, load_chat_model, save_chat_model)

DATA_PATH = ROOT / "data" / "chat" / "security_dialogues.jsonl"


def read_records(path: Path) -> list[dict]:
    records, seen = [], set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("record must be an object")
            if "text" in row:
                if not isinstance(row["text"], str) or not row["text"].strip():
                    raise ValueError("text must be nonempty")
                canonical = {"text": row["text"]}
            else:
                messages = row.get("messages")
                if messages is None:
                    messages = [{"role": "user", "content": row["user"]},
                                {"role": "assistant", "content": row["assistant"]}]
                if not isinstance(messages, list) or not messages or len(messages) % 2:
                    raise ValueError("messages must contain complete user/assistant pairs")
                if any(m.get("role") != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(messages)):
                    raise ValueError("messages must alternate user then assistant")
                encode_messages(messages)
                canonical = {"messages": messages}
            fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
            if fingerprint not in seen:
                records.append({**canonical, "group": str(row.get("group", fingerprint))})
                seen.add(fingerprint)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    if len(records) < 8:
        raise ValueError("Provide at least 8 distinct JSONL records; useful language models need far more.")
    return records


def split_records(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    groups = sorted({r["group"] for r in records})
    if len(groups) < 2:
        raise ValueError("Need at least two conversation/document groups for train and validation.")
    random.Random(seed).shuffle(groups)
    validation = set(groups[:max(1, round(0.1 * len(groups)))])
    return [r for r in records if r["group"] not in validation], [r for r in records if r["group"] in validation]


def make_examples(records: list[dict], context_length: int) -> list[tuple[list[int], list[int]]]:
    examples = []
    for record in records:
        if "text" in record:
            tokens = [*record["text"].encode("utf-8"), END]
            targets = tokens.copy()
        else:
            tokens, targets = encode_messages(record["messages"])
        for start in range(0, len(tokens) - 1, context_length // 2):
            x = tokens[start:start + context_length]
            y = targets[start + 1:start + context_length + 1]
            x = x[:len(y)]
            if any(value != -100 for value in y):
                examples.append((x, y))
            if start + context_length >= len(tokens) - 1:
                break
    if not examples:
        raise ValueError("No trainable text found.")
    return examples


def batch(examples, indices, device):
    chosen = [examples[i] for i in indices]
    length = max(len(x) for x, _ in chosen)
    x = torch.full((len(chosen), length), PAD, dtype=torch.long, device=device)
    y = torch.full_like(x, -100)
    for i, (inputs, targets) in enumerate(chosen):
        x[i, :len(inputs)] = torch.tensor(inputs, device=device)
        y[i, :len(targets)] = torch.tensor(targets, device=device)
    return x, y


@torch.inference_mode()
def validation_loss(model, examples, batch_size, device):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        x, y = batch(examples, range(start, min(start + batch_size, len(examples))), device)
        logits = model(x)
        total += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"))
        count += int((y != -100).sum())
    return total / count


def train(data: Path = DATA_PATH, out: Path = CHAT_MODEL_PATH, *, config: ModelConfig | None = None,
          steps: int = 600, batch_size: int = 16, learning_rate: float = 0.001,
          device: str = "cpu", seed: int = 42, init: Path | None = None, eval_every: int = 100) -> dict:
    if min(steps, batch_size, eval_every) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("steps, batch size, eval interval and learning rate must be positive.")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    records = read_records(data)
    training, validation = split_records(records, seed)
    parent_metadata = None
    if init:
        model, parent_metadata = load_chat_model(init, device)
    else:
        model = ScratchTransformer(config or ModelConfig()).to(device)
    train_examples = make_examples(training, model.config.context_length)
    val_examples = make_examples(validation, model.config.context_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    metadata = {"initialization": "continued_own_checkpoint" if init else "random_weights",
                "data": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "seed": seed, "steps": steps, "parameters": sum(p.numel() for p in model.parameters()),
                "train_records": len(training), "validation_records": len(validation),
                "train_windows": len(train_examples), "validation_windows": len(val_examples),
                "device": device, "learning_rate": learning_rate, "batch_size": batch_size,
                "history": [], "note": "Validation loss is next-byte loss, not a factual-accuracy or security score."}
    if init:
        metadata["parent_checkpoint"] = {"path": str(init), "sha256": hashlib.sha256(init.read_bytes()).hexdigest(),
                                         "initialization": parent_metadata.get("initialization"),
                                         "data": parent_metadata.get("data"), "steps": parent_metadata.get("steps")}
    print(f"{metadata['parameters']:,} parameters | {len(training)} train / {len(validation)} validation records | {device}", flush=True)
    started = time.time()
    best = float("inf")
    for step in range(1, steps + 1):
        model.train()
        x, y = batch(train_examples, [rng.randrange(len(train_examples)) for _ in range(batch_size)], device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss; reduce learning rate and inspect the data.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            val_loss = validation_loss(model, val_examples, batch_size, device)
            record = {"step": step, "train_loss": float(loss.detach()), "validation_loss": val_loss}
            metadata["history"].append(record)
            print(f"step {step}/{steps} | train {record['train_loss']:.3f} | validation {val_loss:.3f} | {time.time()-started:.0f}s", flush=True)
            if val_loss < best:
                best = val_loss
                metadata["best_step"] = step
                metadata["best_validation_loss"] = best
                save_chat_model(model, out, metadata)
    # Retain best validation weights, with the complete run history in the pickle.
    best_model, _ = load_chat_model(out)
    metadata["elapsed_seconds"] = round(time.time() - started, 2)
    save_chat_model(best_model, out, metadata)
    metrics_path = out.with_name(out.stem.replace("_model", "") + "_metrics.json")
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {out} (best step {metadata['best_step']}) and {metrics_path}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=CHAT_MODEL_PATH)
    parser.add_argument("--init", type=Path, help="continue weights from your own trusted pickle; optimizer starts fresh")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    try:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        train(args.data, args.out, config=ModelConfig(args.context_length, args.embed_dim, args.heads, args.layers),
              steps=args.steps, batch_size=args.batch_size, learning_rate=args.learning_rate,
              device=device, seed=args.seed, init=args.init, eval_every=args.eval_every)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
