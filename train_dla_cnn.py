#!/usr/bin/env python3
"""Train a 1D CNN baseline for CSST DLA detection on FITS spectra."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from dla_cnn import (
    DlaCnnModel,
    DlaSpectraDataset,
    build_normalization_stats,
    classification_metrics,
    compute_losses,
    decode_count_predictions,
    denormalize_dla_params,
    load_fits_dataset,
    rmse_masked,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN for CSST DLA spectra.")
    p.add_argument("--train_fits", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=20251031)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--snr_oversample_threshold", type=float, default=1.0)
    p.add_argument("--has_dla_threshold", type=float, default=0.4)
    p.add_argument("--two_dla_threshold", type=float, default=0.5)
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


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate(model: DlaCnnModel,
             loader: DataLoader,
             device: torch.device,
             use_amp: bool,
             loss_config: Dict[str, float],
             has_dla_threshold: float,
             two_dla_threshold: float) -> Dict[str, float]:
    model.eval()
    all_probs = []
    all_two_probs = []
    all_has_dla = []
    all_count_pred = []
    all_count_true = []
    all_params = []
    all_targets = []
    all_masks = []
    all_z_qso = []
    losses = []

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(batch["flux"], batch["aux"])
                loss_dict = compute_losses(outputs, batch, loss_config=loss_config)
            losses.append(float(loss_dict["loss"].item()))
            all_probs.append(torch.sigmoid(outputs["has_dla_logit"]).cpu().numpy())
            all_two_probs.append(torch.sigmoid(outputs["two_dla_logit"]).cpu().numpy())
            all_has_dla.append(batch["has_dla"].cpu().numpy())
            all_count_true.append(batch["n_dla"].cpu().numpy())
            all_params.append(outputs["params_norm"].cpu().numpy())
            all_targets.append(batch["target"].cpu().numpy())
            all_masks.append(batch["mask"].cpu().numpy())
            all_z_qso.append(batch["z_qso"].cpu().numpy())

    probs = np.concatenate(all_probs)
    two_probs = np.concatenate(all_two_probs)
    has_dla = np.concatenate(all_has_dla)
    count_true = np.concatenate(all_count_true)
    params = np.concatenate(all_params)
    targets = np.concatenate(all_targets)
    masks = np.concatenate(all_masks)
    z_qso = np.concatenate(all_z_qso)
    params_phys = denormalize_dla_params(params, z_qso)
    targets_phys = denormalize_dla_params(targets, z_qso)
    count_pred = decode_count_predictions(probs, two_probs, has_dla_threshold, two_dla_threshold)

    metrics = classification_metrics(probs, has_dla)
    metrics["val_loss"] = float(np.mean(losses))
    metrics["count_acc"] = float((count_pred == count_true).mean())
    metrics["z_rmse"] = rmse_masked(params_phys[:, [0, 2]], targets_phys[:, [0, 2]], masks[:, [0, 2]])
    metrics["lognhi_rmse"] = rmse_masked(params_phys[:, [1, 3]], targets_phys[:, [1, 3]], masks[:, [1, 3]])
    snr_gu = loader.dataset.aux[:, 1]
    low = snr_gu < 1.0
    high = ~low
    if np.any(low):
        metrics["f1_low_snr"] = classification_metrics(probs[low], has_dla[low])["f1"]
    else:
        metrics["f1_low_snr"] = float("nan")
    if np.any(high):
        metrics["f1_high_snr"] = classification_metrics(probs[high], has_dla[high])["f1"]
    else:
        metrics["f1_high_snr"] = float("nan")
    return metrics


def build_train_sampler(data: Dict[str, np.ndarray],
                        train_idx: np.ndarray,
                        snr_oversample_threshold: float) -> WeightedRandomSampler:
    has_dla = data["has_dla"][train_idx].astype(np.float32)
    n_dla = data["n_dla"][train_idx].astype(np.int64)
    snr_gu = data["aux"][train_idx, 1].astype(np.float32)
    pos_frac = float(np.mean(has_dla))
    neg_frac = max(1.0 - pos_frac, 1e-3)
    weights = np.ones_like(has_dla, dtype=np.float64)
    weights[has_dla == 1] *= 0.5 / max(pos_frac, 1e-3)
    weights[has_dla == 0] *= 0.5 / neg_frac
    weights[n_dla == 2] *= 1.75
    weights[snr_gu < snr_oversample_threshold] *= 2.0
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(train_idx),
        replacement=True,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = pick_device(args.device)
    use_amp = args.amp and device.type == "cuda"
    print(f"Using device: {device}")

    data = load_fits_dataset(args.train_fits, expect_labels=True)
    n_samples = data["flux"].shape[0]
    indices = np.arange(n_samples)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    n_val = max(1, int(n_samples * args.val_frac))
    val_idx = np.sort(indices[:n_val])
    train_idx = np.sort(indices[n_val:])

    norm_stats = build_normalization_stats(data["aux"], train_idx)

    train_ds = DlaSpectraDataset(data, train_idx, norm_stats, training=True)
    val_ds = DlaSpectraDataset(data, val_idx, norm_stats, training=True)
    train_sampler = build_train_sampler(data, train_idx, args.snr_oversample_threshold)
    pos_weight = float((data["has_dla"][train_idx] == 0).sum() / max((data["has_dla"][train_idx] == 1).sum(), 1))
    loss_config = {
        "has_dla": 1.0,
        "two_dla": 0.35,
        "params": 0.8,
        "pos_weight": pos_weight,
        "focal_alpha": 0.55,
        "focal_gamma": 2.0,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DlaCnnModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    best_f1 = -1.0
    history = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                outputs = model(batch["flux"], batch["aux"])
                loss_dict = compute_losses(outputs, batch, loss_config=loss_config)
            scaler.scale(loss_dict["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss_dict["loss"].item()) * batch["flux"].size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)
        val_metrics = evaluate(
            model, val_loader, device, use_amp,
            loss_config=loss_config,
            has_dla_threshold=args.has_dla_threshold,
            two_dla_threshold=args.two_dla_threshold,
        )
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = train_loss
        history.append(val_metrics)

        print(
            f"Epoch {epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"f1_low={val_metrics['f1_low_snr']:.4f} "
            f"f1_high={val_metrics['f1_high_snr']:.4f} "
            f"count_acc={val_metrics['count_acc']:.4f} "
            f"z_rmse={val_metrics['z_rmse']:.4f} "
            f"lognhi_rmse={val_metrics['lognhi_rmse']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            ckpt = {
                "model_state": model.state_dict(),
                "norm_stats": norm_stats.to_dict(),
                "wave": data["wave"],
                "args": vars(args),
                "metrics": val_metrics,
            }
            torch.save(ckpt, os.path.join(args.output_dir, "best_model.pt"))

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_f1": best_f1,
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "device": str(device),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Best checkpoint: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
