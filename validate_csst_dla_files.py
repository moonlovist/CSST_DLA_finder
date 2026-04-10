#!/usr/bin/env python3
"""Validate train/test/submission FITS files against the challenge schema."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from astropy.io import fits


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_train(path: str) -> None:
    with fits.open(path, memmap=False) as hdul:
        names = [hdu.name for hdu in hdul]
        require("WAVELENGTH" in names, "Missing WAVELENGTH extension")
        require("FLUX" in names, "Missing FLUX extension")
        require("LABELS" in names, "Missing LABELS extension")

        wave = hdul["WAVELENGTH"].data
        flux = hdul["FLUX"].data
        labels = hdul["LABELS"].data
        require(wave.ndim == 1, "WAVELENGTH must be 1D")
        require(flux.ndim == 2, "FLUX must be 2D")
        require(flux.shape[0] == len(labels), "FLUX rows must match LABELS rows")
        require(flux.shape[1] == wave.shape[0], "FLUX wavelength axis must match WAVELENGTH length")
        for name in ["Z_QSO", "HAS_DLA", "N_DLA", "Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2"]:
            require(name in labels.names, f"Missing LABELS field: {name}")


def validate_test(path: str) -> None:
    with fits.open(path, memmap=False) as hdul:
        names = [hdu.name for hdu in hdul]
        require("WAVELENGTH" in names, "Missing WAVELENGTH extension")
        require("FLUX" in names, "Missing FLUX extension")
        require("META" in names, "Missing META extension")

        wave = hdul["WAVELENGTH"].data
        flux = hdul["FLUX"].data
        meta = hdul["META"].data
        require(wave.ndim == 1, "WAVELENGTH must be 1D")
        require(flux.ndim == 2, "FLUX must be 2D")
        require(flux.shape[0] == len(meta), "FLUX rows must match META rows")
        require(flux.shape[1] == wave.shape[0], "FLUX wavelength axis must match WAVELENGTH length")
        for name in ["TARGETID", "Z_QSO"]:
            require(name in meta.names, f"Missing META field: {name}")


def validate_submission(path: str) -> None:
    with fits.open(path, memmap=False) as hdul:
        names = [hdu.name for hdu in hdul]
        require("META" in names, "Submission must contain META")
        require("RESULTS" in names, "Submission must contain RESULTS")
        meta = hdul["META"].data
        results = hdul["RESULTS"].data
        for name in ["TARGETID", "Z_QSO", "Z_DLA", "LOG_NHI", "CONFIDENCE"]:
            require(name in results.names, f"Missing RESULTS field: {name}")
        targetids = set(np.asarray(meta["TARGETID"], dtype=np.int64).tolist())
        result_targetids = np.asarray(results["TARGETID"], dtype=np.int64)
        require(np.all(np.isin(result_targetids, list(targetids))), "RESULTS contains unknown TARGETID")
        conf = np.asarray(results["CONFIDENCE"], dtype=np.float32)
        require(np.all((conf >= 0.0) & (conf <= 1.0)), "CONFIDENCE must lie in [0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate challenge FITS files.")
    parser.add_argument("--train_fits", type=str)
    parser.add_argument("--test_fits", type=str)
    parser.add_argument("--submission_fits", type=str)
    args = parser.parse_args()

    try:
        if args.train_fits:
            validate_train(args.train_fits)
            print(f"Train FITS OK: {args.train_fits}")
        if args.test_fits:
            validate_test(args.test_fits)
            print(f"Test FITS OK: {args.test_fits}")
        if args.submission_fits:
            validate_submission(args.submission_fits)
            print(f"Submission FITS OK: {args.submission_fits}")
    except AssertionError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
