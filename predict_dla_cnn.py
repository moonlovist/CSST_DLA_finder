#!/usr/bin/env python3
"""Run CNN inference on the blind CSST DLA test set."""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader

from dla_cnn import DlaCnnModel, DlaSpectraDataset, NormalizationStats, load_fits_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict DLA labels on a FITS test set.")
    p.add_argument("--test_fits", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def postprocess_params(params_row: np.ndarray, z_qso: float, pred_n_dla: int):
    z1, logn1, z2, logn2 = [float(x) for x in params_row]
    z_min = 1.6
    z_max = max(z_min + 1e-3, float(z_qso) - 0.05)
    z1 = float(np.clip(z1, z_min, z_max))
    z2 = float(np.clip(z2, z_min, z_max))
    logn1 = float(np.clip(logn1, 19.5, 22.5))
    logn2 = float(np.clip(logn2, 19.5, 22.5))

    if pred_n_dla >= 2 and z2 < z1:
        z1, z2 = z2, z1
        logn1, logn2 = logn2, logn1

    if pred_n_dla < 1:
        z1 = np.nan
        logn1 = np.nan
    if pred_n_dla < 2:
        z2 = np.nan
        logn2 = np.nan
    return z1, logn1, z2, logn2


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    norm_stats = NormalizationStats(
        aux_mean=np.asarray(checkpoint["norm_stats"]["aux_mean"], dtype=np.float32),
        aux_std=np.asarray(checkpoint["norm_stats"]["aux_std"], dtype=np.float32),
    )

    data = load_fits_dataset(args.test_fits, expect_labels=False)
    indices = np.arange(data["flux"].shape[0])
    dataset = DlaSpectraDataset(data, indices, norm_stats, training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = DlaCnnModel().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = []
    with torch.no_grad():
        for start, batch in enumerate(loader):
            flux = batch["flux"].to(device, non_blocking=True)
            aux = batch["aux"].to(device, non_blocking=True)
            outputs = model(flux, aux)
            probs = torch.sigmoid(outputs["has_dla_logit"]).cpu().numpy()
            count = outputs["count_logit"].argmax(dim=1).cpu().numpy()
            params = outputs["params"].cpu().numpy()

            for j in range(probs.shape[0]):
                spec_id = start * args.batch_size + j
                pred_n_dla = int(count[j])
                target_id = int(data.get("targetid", np.arange(len(indices)))[spec_id])
                z_qso = float(data.get("z_qso", data["aux"][:, 0])[spec_id])
                z1, logn1, z2, logn2 = postprocess_params(params[j], z_qso, pred_n_dla)
                rows.append([
                    spec_id,
                    target_id,
                    z_qso,
                    float(probs[j]),
                    pred_n_dla,
                    z1,
                    logn1,
                    z2,
                    logn2,
                ])

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "spec_id",
            "targetid",
            "z_qso",
            "prob_has_dla",
            "pred_n_dla",
            "pred_z_dla1",
            "pred_lognhi1",
            "pred_z_dla2",
            "pred_lognhi2",
        ])
        writer.writerows(rows)

    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
