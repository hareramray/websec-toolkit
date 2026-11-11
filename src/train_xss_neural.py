"""Train the neural XSS detector on the GPU, evaluate it, and compare it with the linear model.

Usage:
  python src/train_xss_neural.py [--epochs 12] [--device auto]
  python src/train_xss_neural.py --resume models/xss_neural.pkl --epochs 4 --learning-rate 2e-4

Writes models/xss_neural.pkl and models/xss_neural_metrics.json. One epoch takes
about 20 s on a laptop RTX 5050 on mains power, and several times longer on battery.

With --resume, training continues your own saved checkpoint (same architecture, fresh
optimizer) instead of starting from random weights -- for example to fine-tune the
model on added data. Use a smaller --learning-rate and fewer --epochs than a run from
scratch. The threshold and calibration are always re-tuned on validation data after.

The rows, the train/validation/test split, the threshold rule and the evaluations
are the ones train_xss.py uses, so the two models can be compared number for number:
  1. Held-out test split.
  2. Challenge set (a regression check; see xss_augment.py).
  3. External benchmark from prepare_external.py, which nothing is fitted or tuned on.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_xss import DATA_PATH, by_source, challenge, evaluate, external, show, split_data, threshold_for_fpr  # noqa: E402
from xss_augment import epoch_examples  # noqa: E402
from xss_model import MODEL_PATH, WINDOW, load_model, score  # noqa: E402
from xss_neural import NEURAL_PATH, NeuralConfig, NeuralXSS, XSSNet, encode, load_xssnet, pad, save_neural  # noqa: E402

SEED = 42
BUCKET = 50  # batches sorted by length together, so short rows are not padded to 1000
LONG_ROW = 250  # rows up to this many characters use the full batch size; longer rows use proportionally less
LONG_BENIGN_WINDOWS = 1  # random windows taken from each long benign text per epoch


def batches(rows: list[np.ndarray], labels: list[int], batch_size: int, rng: random.Random) -> list[tuple[list[np.ndarray], list[int]]]:
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = []
    for start in range(0, len(order), batch_size * BUCKET):
        group = sorted(order[start : start + batch_size * BUCKET], key=lambda i: len(rows[i]))
        i = 0
        while i < len(group):
            # Attention memory grows with length squared, so long rows go in smaller batches.
            size = max(8, min(batch_size, batch_size * LONG_ROW // len(rows[group[min(i + batch_size, len(group)) - 1]])))
            out.append(group[i : i + size])
            i += size
    rng.shuffle(out)
    return [([rows[i] for i in picks], [labels[i] for i in picks]) for picks in out]


def epoch_rows(df: pd.DataFrame, rng: random.Random) -> tuple[list[str], list[int]]:
    """This epoch's base rows, cut to the window the model scores at.

    A long benign text contributes windows from random positions, new ones every
    epoch, so the model sees the middle of large harmless files without paying for
    all of them each time. An attack contributes only its first window: a later
    window may not contain the payload and would be mislabelled.
    """
    texts, labels = [], []
    for text, label in zip(df["text"], df["label"]):
        if label == 1 or len(text) <= WINDOW:
            texts.append(text[:WINDOW])
            labels.append(int(label))
        else:
            for _ in range(LONG_BENIGN_WINDOWS):
                start = rng.randrange(len(text) - WINDOW + 1)
                texts.append(text[start : start + WINDOW])
                labels.append(0)
    return texts, labels


def fit_neural(train: pd.DataFrame, val: pd.DataFrame | None = None, *, config: NeuralConfig | None = None, epochs: int = 12,
               batch_size: int = 128, learning_rate: float = 6e-4, device: str = "cpu", seed: int = SEED,
               init_model: XSSNet | None = None, log=print) -> tuple[XSSNet, list[dict]]:
    """Train the network. Returns the averaged weights of the best epoch and the history.

    Starts from random weights, or from `init_model` to continue your own checkpoint
    (its architecture is used; the optimizer always starts fresh). With `val`, the best
    epoch is the one with the highest validation detection at 0.1% false alarms, checked
    in the second half of training. Without it, the last epoch is returned.
    """
    if min(epochs, batch_size) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive.")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = (init_model if init_model is not None else XSSNet(config or NeuralConfig())).to(device)
    average = copy.deepcopy(model).eval()  # running average of the weights: steadier than the last step
    for parameter in average.parameters():
        parameter.requires_grad_(False)
    decay = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05}, {"params": plain, "weight_decay": 0.0}], lr=learning_rate)
    use_amp = str(device).startswith("cuda")

    history, best, best_state, step, total = [], None, None, 0, 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        epoch_texts, epoch_labels = epoch_examples(*epoch_rows(train, rng), rng)
        rows = encode(epoch_texts, model.config.max_length)
        epoch_batches = batches(rows, epoch_labels, batch_size, rng)
        total = total or epochs * len(epoch_batches)
        warmup = max(1, int(0.05 * total))
        model.train()
        loss_sum, seen = 0.0, 0
        for batch_rows, batch_labels in epoch_batches:
            progress = min(1.0, step / total)
            rate = learning_rate * (step + 1) / warmup if step < warmup else learning_rate * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = rate
            x = pad(batch_rows, device)
            y = torch.tensor(batch_labels, dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; reduce the learning rate.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            keep = min(0.999, (1 + step) / (10 + step))
            with torch.no_grad():
                for old, new in zip(average.parameters(), model.parameters()):
                    old.lerp_(new, 1 - keep)
            loss_sum = loss_sum + loss.detach() * len(batch_labels)  # stays on the GPU: no wait per step
            seen += len(batch_labels)

        record = {"epoch": epoch, "train_loss": round(float(loss_sum) / seen, 5), "rows": seen}
        if val is not None and (epoch > epochs // 2 or epoch == epochs):  # early epochs are never the best; skip their check
            y_val = val["label"].to_numpy()
            scores = score(NeuralXSS(average, device=device, fast=True), val["text"].tolist(), normalized=True)
            record["validation_detection_at_0.1pct_false_alarms"] = round(float((scores[y_val == 1] > threshold_for_fpr(y_val, scores, 0.001)).mean()), 4)
            record["validation_pr_auc"] = round(float(average_precision_score(y_val, scores)), 5)
            rank = (record["validation_detection_at_0.1pct_false_alarms"], record["validation_pr_auc"])
            if best is None or rank > best:
                best, best_state = rank, copy.deepcopy(average.state_dict())
                record["best"] = True
        record["seconds"] = round(time.time() - started, 1)
        history.append(record)
        log("  " + " | ".join(f"{key} {value}" for key, value in record.items()), flush=True)
    if best_state is not None:
        average.load_state_dict(best_state)
    return average, history


def calibrate(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that makes the scores honest probabilities on validation data.

    Dividing every logit by one number changes no ranking and no detection rate. It
    only spreads out scores that the network pushes to exactly 0 or 1.
    """
    def loss(temperature: float) -> float:
        z = logits / temperature
        return float(np.mean(np.logaddexp(0, z) - y * z))
    return float(min(np.geomspace(1, 30, 80), key=loss))


def to_logits(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, 1e-300, 1 - 1e-16)
    return np.log(p) - np.log1p(-p)


def compare_with(baseline: Path, test: pd.DataFrame, known: set[str], neural: dict) -> dict | None:
    """Score the saved linear model on the same test rows, challenge set and external benchmark."""
    if not baseline.exists():
        return None
    bundle = load_model(baseline)
    if bundle.get("kind"):
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        linear = {"held_out_test": evaluate(test["label"].to_numpy(), score(bundle["pipeline"], test["text"].tolist(), normalized=True), bundle["threshold"]),
                  "challenge_set": challenge(bundle["pipeline"], bundle["threshold"]),
                  "external_benchmark": external(bundle["pipeline"], bundle["threshold"], known)}

    def row(name: str, pick) -> None:
        try:
            print(f"  {name:<46} {pick(linear):>14} {pick(neural):>14}")
        except (KeyError, TypeError):
            pass

    print(f"\nSame rows, each model at its own validation-tuned threshold\n  {'':<46} {'linear':>14} {'neural':>14}")
    row("held-out detection rate", lambda m: f"{m['held_out_test']['detection_rate']:.2%}")
    row("held-out false-alarm rate", lambda m: f"{m['held_out_test']['false_alarm_rate']:.2%}")
    row("held-out detection at 0.1% false alarms", lambda m: f"{m['held_out_test']['detection_at_0.1pct_false_alarms']:.2%}")
    row("held-out attacks missed", lambda m: m["held_out_test"]["confusion"]["fn"])
    row("challenge attacks caught", lambda m: f"{m['challenge_set']['caught']}/{m['challenge_set']['attacks']}")
    row("challenge benign false alarms", lambda m: f"{len(m['challenge_set']['false_alarms'])}/{m['challenge_set']['benign']}")
    row("external attacks caught", lambda m: f"{m['external_benchmark']['caught']}/{m['external_benchmark']['attacks']}")
    row("external attacks caught (not in project data)", lambda m: f"{m['external_benchmark']['unseen_caught']}/{m['external_benchmark']['unseen_attacks']}")
    row("external benign false alarms", lambda m: f"{m['external_benchmark']['false_alarms']}/{m['external_benchmark']['benign']}")
    return {"path": str(baseline), **linear}


def train(data: Path = DATA_PATH, out: Path = NEURAL_PATH, *, target_fpr: float = 0.001, epochs: int = 12, batch_size: int = 128,
          learning_rate: float = 6e-4, config: NeuralConfig | None = None, device: str = "auto", seed: int = SEED,
          baseline: Path = MODEL_PATH, exclude_sources: tuple[str, ...] = (), init: Path | None = None) -> dict:
    """Train on `data`, evaluate, save the pickle and the metrics. Returns the metrics.

    `baseline` must have been trained with the same `exclude_sources`, or its scores
    on this test split are not a fair comparison. With `init`, training continues your
    own saved neural checkpoint (same architecture, fresh optimizer) instead of starting
    from random weights; the threshold and calibration are re-tuned afterwards as usual.
    """
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be between 0 and 1")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    df, cleaning, train_rows, val, test = split_data(data, exclude_sources)

    init_model, parent = None, None
    if init:
        init_model, parent_bundle = load_xssnet(init, "cpu")
        config = NeuralConfig(**parent_bundle["config"])  # the checkpoint's architecture wins; weights must fit it
        parent = parent_bundle.get("metadata") or {}
        print(f"continuing {init} ({sum(p.numel() for p in init_model.parameters()):,} parameters); optimizer starts fresh")
    config = config or NeuralConfig()
    started = time.time()
    print(f"training on {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.startswith("cuda") else ""), flush=True)
    model, history = fit_neural(train_rows, val, config=config, epochs=epochs, batch_size=batch_size,
                                learning_rate=learning_rate, device=device, seed=seed, init_model=init_model)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"trained in {time.time() - started:.0f}s, {parameters:,} parameters")

    # Calibrate and tune the threshold on validation data, never on the test set.
    y_val = val["label"].to_numpy()
    temperature = calibrate(to_logits(score(NeuralXSS(model, device=device), val["text"].tolist(), normalized=True)), y_val)
    detector = NeuralXSS(model, temperature, device)
    threshold = max(0.5, threshold_for_fpr(y_val, score(detector, val["text"].tolist(), normalized=True), target_fpr))

    best = max((record["epoch"] for record in history if record.get("best")), default=epochs)
    model_info = {"kind": "char_cnn_transformer", "initialization": "continued_own_checkpoint" if init else "random_weights",
                  "parameters": parameters, "config": asdict(config), "device": device, "epochs": epochs, "best_epoch": best,
                  "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed, "temperature": round(temperature, 4),
                  "training_seconds": round(time.time() - started, 1), "torch_version": str(torch.__version__)}
    if init:
        parent_model = (parent or {}).get("model", {})
        model_info["parent_checkpoint"] = {"path": str(init), "sha256": hashlib.sha256(init.read_bytes()).hexdigest(),
                                           "initialization": parent_model.get("initialization"), "rows": (parent or {}).get("rows"),
                                           "best_epoch": parent_model.get("best_epoch"), "torch_version": parent_model.get("torch_version")}
    metrics = {"model": model_info,
               "threshold": threshold, "target_fpr": target_fpr, "rows": int(len(df)), "sources": sorted(df["source"].unique()),
               "cleaning": cleaning, "history": history}
    test_scores = score(detector, test["text"].tolist(), normalized=True)
    metrics["held_out_test"] = evaluate(test["label"].to_numpy(), test_scores, threshold)
    show("Held-out test split", metrics["held_out_test"])
    metrics["held_out_test"]["by_source"] = by_source(test, test_scores, threshold)

    known = set(df["text"])
    metrics["challenge_set"] = challenge(detector, threshold)
    metrics["external_benchmark"] = external(detector, threshold, known)
    metrics["linear_baseline"] = compare_with(baseline, test, known, metrics)

    save_neural(model, threshold, temperature, out, {key: metrics[key] for key in ("model", "threshold", "target_fpr", "rows", "sources")})
    metrics_path = out.with_name(out.stem.replace("_model", "") + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nsaved {out} and {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, default=NEURAL_PATH)
    parser.add_argument("--baseline", type=Path, default=MODEL_PATH, help="linear model pickle to compare against")
    parser.add_argument("--target-fpr", type=float, default=0.001, help="false-alarm rate the threshold is tuned for")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--exclude-source", action="append", default=[],
                        help="drop rows of this source before splitting (repeatable); train --baseline with the same exclusions")
    parser.add_argument("--resume", type=Path,
                        help="continue your own trusted neural pickle with a fresh optimizer; its architecture is reused, so --width/--layers/--heads are ignored")
    args = parser.parse_args()
    if not args.data.exists():
        sys.exit(f"no data at {args.data}: run  python src/prepare_data.py  first")
    if args.resume and not args.resume.exists():
        sys.exit(f"no checkpoint at {args.resume} to resume from")
    try:
        train(args.data, args.out, target_fpr=args.target_fpr, epochs=args.epochs, batch_size=args.batch_size,
              learning_rate=args.learning_rate, config=NeuralConfig(width=args.width, layers=args.layers, heads=args.heads, dropout=args.dropout),
              device=args.device, seed=args.seed, baseline=args.baseline, exclude_sources=tuple(args.exclude_source), init=args.resume)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
