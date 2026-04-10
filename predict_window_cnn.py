#!/usr/bin/env python3
"""Run sliding-window CNN inference and write a debugging CSV."""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from window_cnn import (
    LYA_REST,
    LOGNHI_MAX,
    LOGNHI_MIN,
    WindowCnn,
    extract_window,
    load_test_arrays,
    merge_candidates,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_fits", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--window_size", type=int, default=256)
    p.add_argument("--confidence_threshold", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=8)
    return p.parse_args()


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_spectrum(model, wave, flux, z_qso, device, window_size, stride, threshold, top_k):
    half = window_size // 2
    candidates = []
    model.eval()
    with torch.no_grad():
        for center in range(half, len(wave) - half, stride):
            window = extract_window(flux, center, window_size)[None, None, :]
            aux = np.array([[wave[center] / 8000.0, z_qso / 4.0]], dtype=np.float32)
            outputs = model(torch.from_numpy(window).to(device), torch.from_numpy(aux).to(device))
            conf = float(torch.sigmoid(outputs["logit"])[0].cpu().item())
            if conf < threshold:
                continue
            offset = float(outputs["offset"][0].cpu().item())
            refined_center = center + offset * 6.0
            refined_center = float(np.clip(refined_center, 0, len(wave) - 1))
            lam = np.interp(refined_center, np.arange(len(wave), dtype=np.float32), wave)
            z_dla = lam / LYA_REST - 1.0
            logn = LOGNHI_MIN + float(outputs["logn_norm"][0].cpu().item()) * (LOGNHI_MAX - LOGNHI_MIN)
            if z_dla >= z_qso - 0.05 or z_dla < 1.6:
                continue
            candidates.append({
                "center_pix": refined_center,
                "z_dla": z_dla,
                "log_nhi": logn,
                "confidence": conf,
            })
    merged = merge_candidates(candidates)
    merged = sorted(merged, key=lambda x: x["confidence"], reverse=True)[:top_k]
    merged.sort(key=lambda x: x["z_dla"])
    return merged


def main():
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = WindowCnn().to(device)
    model.load_state_dict(checkpoint["model_state"])
    data = load_test_arrays(args.test_fits)
    rows = []
    for i in range(len(data["flux"])):
        cands = infer_spectrum(
            model, data["wave"], data["flux"][i], float(data["z_qso"][i]), device,
            args.window_size, args.stride, args.confidence_threshold, args.top_k,
        )
        if not cands:
            rows.append([i, int(data["targetid"][i]), float(data["z_qso"][i]), "", "", ""])
            continue
        for cand in cands:
            rows.append([
                i,
                int(data["targetid"][i]),
                float(data["z_qso"][i]),
                float(cand["confidence"]),
                float(cand["z_dla"]),
                float(cand["log_nhi"]),
            ])
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["spec_id", "targetid", "z_qso", "confidence", "pred_z_dla", "pred_log_nhi"])
        writer.writerows(rows)
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
