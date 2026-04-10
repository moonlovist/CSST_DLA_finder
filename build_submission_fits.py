#!/usr/bin/env python3
"""Build the official CSST DLA submission FITS with a RESULTS extension."""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from torch.utils.data import DataLoader

from dla_cnn import DlaCnnModel, DlaSpectraDataset, NormalizationStats, load_fits_dataset
from predict_dla_cnn import pick_device, postprocess_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build submission FITS for the CSST DLA challenge.")
    p.add_argument("--test_fits", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_fits", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--confidence_threshold", type=float, default=0.5)
    p.add_argument("--min_log_nhi", type=float, default=20.3)
    return p.parse_args()


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

    targetids = []
    z_qsos = []
    z_dlas = []
    log_nhis = []
    confidences = []

    with torch.no_grad():
        for start, batch in enumerate(loader):
            flux = batch["flux"].to(device, non_blocking=True)
            aux = batch["aux"].to(device, non_blocking=True)
            outputs = model(flux, aux)
            probs = torch.sigmoid(outputs["has_dla_logit"]).cpu().numpy()
            counts = outputs["count_logit"].argmax(dim=1).cpu().numpy()
            params = outputs["params"].cpu().numpy()

            for j in range(probs.shape[0]):
                spec_id = start * args.batch_size + j
                confidence = float(probs[j])
                if confidence < args.confidence_threshold:
                    continue

                pred_n_dla = int(counts[j])
                if pred_n_dla < 1:
                    continue

                targetid = int(data["targetid"][spec_id])
                z_qso = float(data["z_qso"][spec_id])
                z1, logn1, z2, logn2 = postprocess_params(params[j], z_qso, pred_n_dla)

                rows = []
                if np.isfinite(z1) and np.isfinite(logn1) and logn1 >= args.min_log_nhi:
                    rows.append((z1, logn1, confidence))
                if np.isfinite(z2) and np.isfinite(logn2) and logn2 >= args.min_log_nhi:
                    rows.append((z2, logn2, confidence))

                for z_dla, log_nhi, conf in rows:
                    targetids.append(targetid)
                    z_qsos.append(z_qso)
                    z_dlas.append(float(z_dla))
                    log_nhis.append(float(log_nhi))
                    confidences.append(float(conf))

    results = Table({
        "TARGETID": np.asarray(targetids, dtype=np.int64),
        "Z_QSO": np.asarray(z_qsos, dtype=np.float32),
        "Z_DLA": np.asarray(z_dlas, dtype=np.float32),
        "LOG_NHI": np.asarray(log_nhis, dtype=np.float32),
        "CONFIDENCE": np.asarray(confidences, dtype=np.float32),
    })
    if len(results) > 0:
        results.sort(["TARGETID", "Z_DLA"])

    shutil.copy(args.test_fits, args.output_fits)
    with fits.open(args.output_fits, mode="append") as hdul:
        if "RESULTS" in hdul:
            raise RuntimeError("Output FITS already contains a RESULTS extension.")
        hdu = fits.BinTableHDU(results.as_array(), name="RESULTS")
        hdu.header["COMMENT"] = "One row per predicted DLA absorber."
        hdul.append(hdu)

    print(f"Wrote submission FITS to {args.output_fits}")
    print(f"Number of predicted DLAs: {len(results)}")


if __name__ == "__main__":
    main()
