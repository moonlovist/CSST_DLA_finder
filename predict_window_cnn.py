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
    candidate_rank,
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
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--merge_separation_pix", type=int, default=80)
    p.add_argument(
        "--rank_by",
        choices=["confidence", "logn", "conf_logn", "support", "conf_support", "mean_conf", "cluster_score"],
        default="confidence",
    )
    p.add_argument("--min_support", type=float, default=1.0)
    return p.parse_args()


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer_spectrum(model, wave, flux, z_qso, device, window_size, stride, threshold, top_k,
                   batch_size, merge_separation_pix=80, offset_scale_pix=16, rank_by="confidence",
                   min_support=1.0):
    half = window_size // 2
    candidates = []
    model.eval()
    centers = list(range(half, len(wave) - half, stride))
    with torch.no_grad():
        for i in range(0, len(centers), batch_size):
            batch_centers = centers[i:i + batch_size]
            windows = np.stack([extract_window(flux, c, window_size) for c in batch_centers], axis=0)[:, None, :]
            aux = np.stack(
                [np.array([wave[c] / 8000.0, z_qso / 4.0], dtype=np.float32) for c in batch_centers],
                axis=0,
            )
            outputs = model(torch.from_numpy(windows).to(device), torch.from_numpy(aux).to(device))
            confs = torch.sigmoid(outputs["logit"]).cpu().numpy()
            offsets = outputs["offset"].cpu().numpy()
            logns = outputs["logn_norm"].cpu().numpy()
            for center, conf, offset, logn_norm in zip(batch_centers, confs, offsets, logns):
                conf = float(conf)
                if conf < threshold:
                    continue
                refined_center = center + float(offset) * max(offset_scale_pix, 1)
                refined_center = float(np.clip(refined_center, 0, len(wave) - 1))
                lam = np.interp(refined_center, np.arange(len(wave), dtype=np.float32), wave)
                z_dla = lam / LYA_REST - 1.0
                logn = LOGNHI_MIN + float(logn_norm) * (LOGNHI_MAX - LOGNHI_MIN)
                if z_dla >= z_qso - 0.05 or z_dla < 1.6:
                    continue
                candidates.append({
                    "center_pix": refined_center,
                    "z_dla": z_dla,
                    "log_nhi": logn,
                    "confidence": conf,
                })
    merged = merge_candidates(candidates, min_separation_pix=merge_separation_pix, rank_by=rank_by)
    merged = [cand for cand in merged if float(cand.get("support", 1.0)) >= min_support]
    merged = sorted(merged, key=lambda x: candidate_rank(x, rank_by), reverse=True)[:top_k]
    merged.sort(key=lambda x: x["z_dla"])
    return merged


def main():
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = checkpoint.get("args", {})
    offset_scale_pix = int(train_args.get("positive_radius_pix", train_args.get("jitter_pix", 16)))
    model = WindowCnn().to(device)
    model.load_state_dict(checkpoint["model_state"])
    data = load_test_arrays(args.test_fits)
    rows = []
    for i in range(len(data["flux"])):
        cands = infer_spectrum(
            model, data["wave"], data["flux"][i], float(data["z_qso"][i]), device,
            args.window_size, args.stride, args.confidence_threshold, args.top_k, args.batch_size,
            args.merge_separation_pix, offset_scale_pix, args.rank_by, args.min_support,
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
                float(cand.get("support", 1.0)),
                float(cand.get("mean_conf", cand["confidence"])),
                float(cand.get("sum_conf", cand["confidence"])),
                float(cand.get("width_pix", 0.0)),
            ])
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "spec_id", "targetid", "z_qso", "confidence", "pred_z_dla", "pred_log_nhi",
            "support", "mean_conf", "sum_conf", "width_pix",
        ])
        writer.writerows(rows)
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
