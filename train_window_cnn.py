#!/usr/bin/env python3
"""Train a DESI-style window CNN baseline for CSST DLA finding."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from window_cnn import (
    LOGNHI_MAX,
    LOGNHI_MIN,
    WindowCnn,
    WindowSpectraDataset,
    build_window_samples,
    compute_window_loss,
    load_train_arrays,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_fits", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--window_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20251031)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate(model, loader, device, use_amp, pos_weight):
    model.eval()
    losses = []
    probs = []
    labels = []
    pred_logn = []
    true_logn = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(batch["window"], batch["aux"])
                loss_dict = compute_window_loss(outputs, batch, pos_weight=pos_weight)
            losses.append(float(loss_dict["loss"].item()))
            probs.append(torch.sigmoid(outputs["logit"]).cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            pred_logn.append(outputs["logn_norm"].cpu().numpy())
            true_logn.append(batch["logn"].cpu().numpy())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    pred_logn = np.concatenate(pred_logn)
    true_logn = np.concatenate(true_logn)
    preds = (probs >= 0.5).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    pos = labels == 1
    if np.any(pos):
        pred_phys = LOGNHI_MIN + pred_logn[pos] * (LOGNHI_MAX - LOGNHI_MIN)
        true_phys = LOGNHI_MIN + true_logn[pos] * (LOGNHI_MAX - LOGNHI_MIN)
        logn_rmse = float(np.sqrt(np.mean((pred_phys - true_phys) ** 2)))
    else:
        logn_rmse = float("nan")
    return {
        "val_loss": float(np.mean(losses)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "logn_rmse": logn_rmse,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = pick_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    print(f"Using device: {device}")

    data = load_train_arrays(args.train_fits)
    n = len(data["flux"])
    idx = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * args.val_frac))
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])

    train_samples = build_window_samples(data, train_idx, args.window_size, seed=args.seed)
    val_samples = build_window_samples(data, val_idx, args.window_size, seed=args.seed + 1)

    train_ds = WindowSpectraDataset(data, train_samples, args.window_size)
    val_ds = WindowSpectraDataset(data, val_samples, args.window_size)
    pos_weight = max(sum(s.label == 0 for s in train_samples) / max(sum(s.label == 1 for s in train_samples), 1), 1.0)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = WindowCnn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(batch["window"], batch["aux"])
                loss_dict = compute_window_loss(outputs, batch, pos_weight=pos_weight)
            scaler.scale(loss_dict["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss_dict["loss"].item()) * batch["window"].size(0)

        scheduler.step()
        train_loss = running / len(train_ds)
        metrics = evaluate(model, val_loader, device, use_amp, pos_weight)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)
        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} f1={metrics['f1']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
            f"lognhi_rmse={metrics['logn_rmse']:.4f}"
        )
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_f1": best_f1, "n_train_windows": len(train_ds), "n_val_windows": len(val_ds)}, f, indent=2)
    print(f"Best checkpoint: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
