#!/usr/bin/env python3
"""Build official submission FITS from the window-CNN detector."""

from __future__ import annotations

import argparse

import numpy as np
from astropy.io import fits
from astropy.table import Table

from predict_window_cnn import infer_spectrum, pick_device
from window_cnn import WindowCnn, load_test_arrays
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_fits", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_fits", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--window_size", type=int, default=256)
    p.add_argument("--confidence_threshold", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--min_log_nhi", type=float, default=20.3)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WindowCnn().to(device)
    model.load_state_dict(checkpoint["model_state"])
    data = load_test_arrays(args.test_fits)

    targetids = []
    z_qsos = []
    z_dlas = []
    log_nhis = []
    confidences = []

    for i in range(len(data["flux"])):
        cands = infer_spectrum(
            model, data["wave"], data["flux"][i], float(data["z_qso"][i]), device,
            args.window_size, args.stride, args.confidence_threshold, args.top_k,
        )
        for cand in cands:
            if cand["log_nhi"] < args.min_log_nhi:
                continue
            targetids.append(int(data["targetid"][i]))
            z_qsos.append(float(data["z_qso"][i]))
            z_dlas.append(float(cand["z_dla"]))
            log_nhis.append(float(cand["log_nhi"]))
            confidences.append(float(cand["confidence"]))

    results = Table({
        "TARGETID": np.asarray(targetids, dtype=np.int64),
        "Z_QSO": np.asarray(z_qsos, dtype=np.float32),
        "Z_DLA": np.asarray(z_dlas, dtype=np.float32),
        "LOG_NHI": np.asarray(log_nhis, dtype=np.float32),
        "CONFIDENCE": np.asarray(confidences, dtype=np.float32),
    })
    if len(results) > 0:
        results.sort(["TARGETID", "Z_DLA"])

    with fits.open(args.test_fits, memmap=False) as hdul:
        hdus = [hdu.copy() for hdu in hdul]
    hdus.append(fits.BinTableHDU(results.as_array(), name="RESULTS"))
    fits.HDUList(hdus).writeto(args.output_fits, overwrite=True)
    print(f"Wrote submission FITS to {args.output_fits}")
    print(f"Number of predicted DLAs: {len(results)}")


if __name__ == "__main__":
    main()
