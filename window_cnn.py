#!/usr/bin/env python3
"""Window-based CNN utilities for CSST DLA detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from torch.utils.data import Dataset


LYA_REST = 1215.67
LOGNHI_MIN = 19.5
LOGNHI_MAX = 22.5


def load_train_arrays(path: str) -> Dict[str, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float32)
        flux = np.asarray(hdul["FLUX"].data, dtype=np.float32)
        labels = hdul["LABELS"].data
        data = {
            "wave": wave,
            "flux": flux,
            "z_qso": np.asarray(labels["Z_QSO"], dtype=np.float32),
            "has_dla": np.asarray(labels["HAS_DLA"], dtype=np.int64),
            "n_dla": np.asarray(labels["N_DLA"], dtype=np.int64),
            "z1": np.asarray(labels["Z_DLA1"], dtype=np.float32),
            "logn1": np.asarray(labels["LOGNHI1"], dtype=np.float32),
            "z2": np.asarray(labels["Z_DLA2"], dtype=np.float32),
            "logn2": np.asarray(labels["LOGNHI2"], dtype=np.float32),
            "snr_gu": np.asarray(labels["SNR_GU"], dtype=np.float32),
        }
    return data


def load_test_arrays(path: str) -> Dict[str, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float32)
        flux = np.asarray(hdul["FLUX"].data, dtype=np.float32)
        if "META" in hdul:
            meta = hdul["META"].data
            z_qso = np.asarray(meta["Z_QSO"], dtype=np.float32)
            targetid = np.asarray(meta["TARGETID"], dtype=np.int64)
        elif "LABELS" in hdul:
            # Backward compatibility for older blind-test files produced before META was added.
            meta = hdul["LABELS"].data
            z_qso = np.asarray(meta["Z_QSO"], dtype=np.float32)
            if "TARGETID" in meta.names:
                targetid = np.asarray(meta["TARGETID"], dtype=np.int64)
            else:
                targetid = np.arange(flux.shape[0], dtype=np.int64)
        else:
            raise KeyError("Neither META nor LABELS extension found in test FITS.")
        data = {
            "wave": wave,
            "flux": flux,
            "z_qso": z_qso,
            "targetid": targetid,
        }
    return data


def robust_scale(flux: np.ndarray) -> np.ndarray:
    scale = np.percentile(np.abs(flux), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.float32(scale)


def dla_centers_pix(wave: np.ndarray,
                    z1: float,
                    z2: float) -> List[int]:
    centers = []
    for z in [z1, z2]:
        if np.isfinite(z):
            lam = LYA_REST * (1.0 + float(z))
            centers.append(int(np.argmin(np.abs(wave - lam))))
    centers.sort()
    return centers


def extract_window(flux: np.ndarray, center: int, window_size: int) -> np.ndarray:
    half = window_size // 2
    start = center - half
    end = center + half
    window = np.zeros(window_size, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(len(flux), end)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)
    window[dst_start:dst_end] = flux[src_start:src_end]
    scale = robust_scale(window)
    window = np.clip(window / scale, -5.0, 5.0)
    return window


@dataclass
class WindowSample:
    spec_idx: int
    center_idx: int
    label: int
    offset_norm: float
    logn_norm: float


def build_window_samples(data: Dict[str, np.ndarray],
                         indices: Sequence[int],
                         window_size: int,
                         num_neg_per_spec: int = 6,
                         jitter_pix: int = 16,
                         min_positive_log_nhi: float = 20.3,
                         num_hard_neg_per_dla: int = 4,
                         hard_neg_min_pix: int = 24,
                         seed: int = 20251031) -> List[WindowSample]:
    wave = data["wave"]
    rng = np.random.default_rng(seed)
    half = window_size // 2
    low = half
    high = len(wave) - half - 1
    samples: List[WindowSample] = []

    for spec_idx in indices:
        all_centers = []
        for z, logn in [
            (data["z1"][spec_idx], data["logn1"][spec_idx]),
            (data["z2"][spec_idx], data["logn2"][spec_idx]),
        ]:
            if np.isfinite(z):
                center = int(np.argmin(np.abs(wave - LYA_REST * (1.0 + float(z)))))
                all_centers.append((center, float(logn)))

        positive_centers = [(c, logn) for c, logn in all_centers if np.isfinite(logn) and logn >= min_positive_log_nhi]
        for c, logn in positive_centers:
            for _ in range(3):
                jitter = int(rng.integers(-jitter_pix, jitter_pix + 1))
                anchor = int(np.clip(c + jitter, low, high))
                offset = (c - anchor) / max(jitter_pix, 1)
                logn_norm = (float(logn) - LOGNHI_MIN) / (LOGNHI_MAX - LOGNHI_MIN)
                samples.append(WindowSample(spec_idx, anchor, 1, float(offset), float(logn_norm)))

            for sign in [-1, 1]:
                for _ in range(max(num_hard_neg_per_dla // 2, 1)):
                    dist = int(rng.integers(hard_neg_min_pix, window_size // 2 + 1))
                    anchor = int(np.clip(c + sign * dist, low, high))
                    samples.append(WindowSample(spec_idx, anchor, 0, 0.0, 0.0))

        forbidden = np.zeros(len(wave), dtype=bool)
        for c, _ in all_centers:
            lo = max(0, c - window_size // 2)
            hi = min(len(wave), c + window_size // 2)
            forbidden[lo:hi] = True

        negatives = 0
        tries = 0
        while negatives < num_neg_per_spec and tries < num_neg_per_spec * 20:
            tries += 1
            anchor = int(rng.integers(low, high + 1))
            if forbidden[anchor]:
                continue
            samples.append(WindowSample(spec_idx, anchor, 0, 0.0, 0.0))
            negatives += 1

    return samples


def build_sliding_window_samples(data: Dict[str, np.ndarray],
                                 indices: Sequence[int],
                                 window_size: int,
                                 stride: int = 16,
                                 positive_radius_pix: int = 16,
                                 min_positive_log_nhi: float = 20.3,
                                 max_neg_per_spec: int = 32,
                                 hard_negative_radius_pix: int = 96,
                                 seed: int = 20251031) -> List[WindowSample]:
    """Build training windows from the same grid used at inference time.

    This reduces the train/inference mismatch from random window sampling. Anchors near
    true DLAs above the challenge threshold are positive; anchors near sub-DLAs or just
    outside positive regions are retained preferentially as hard negatives.
    """
    wave = data["wave"]
    rng = np.random.default_rng(seed)
    half = window_size // 2
    low = half
    high = len(wave) - half - 1
    grid_centers = np.arange(low, high + 1, stride, dtype=np.int64)
    samples: List[WindowSample] = []

    for spec_idx in indices:
        absorber_centers = []
        positive_centers = []
        for z, logn in [
            (data["z1"][spec_idx], data["logn1"][spec_idx]),
            (data["z2"][spec_idx], data["logn2"][spec_idx]),
        ]:
            if not (np.isfinite(z) and np.isfinite(logn)):
                continue
            center = int(np.argmin(np.abs(wave - LYA_REST * (1.0 + float(z)))))
            absorber_centers.append((center, float(logn)))
            if float(logn) >= min_positive_log_nhi:
                positive_centers.append((center, float(logn)))

        neg_candidates: List[tuple[int, bool]] = []
        for anchor in grid_centers:
            if positive_centers:
                distances = np.asarray([abs(anchor - c) for c, _ in positive_centers])
                nearest = int(np.argmin(distances))
                if distances[nearest] <= positive_radius_pix:
                    c, logn = positive_centers[nearest]
                    offset = (c - int(anchor)) / max(positive_radius_pix, 1)
                    logn_norm = (float(logn) - LOGNHI_MIN) / (LOGNHI_MAX - LOGNHI_MIN)
                    samples.append(WindowSample(spec_idx, int(anchor), 1, float(offset), float(logn_norm)))
                    continue

            hard = False
            if absorber_centers:
                min_dist_any = min(abs(int(anchor) - c) for c, _ in absorber_centers)
                hard = min_dist_any <= hard_negative_radius_pix
            neg_candidates.append((int(anchor), hard))

        hard_negs = [anchor for anchor, hard in neg_candidates if hard]
        easy_negs = [anchor for anchor, hard in neg_candidates if not hard]
        rng.shuffle(hard_negs)
        rng.shuffle(easy_negs)
        n_hard = min(len(hard_negs), max_neg_per_spec // 2)
        n_easy = max_neg_per_spec - n_hard
        chosen_negs = hard_negs[:n_hard] + easy_negs[:n_easy]
        for anchor in chosen_negs:
            samples.append(WindowSample(spec_idx, int(anchor), 0, 0.0, 0.0))

    return samples


class WindowSpectraDataset(Dataset):
    def __init__(self,
                 data: Dict[str, np.ndarray],
                 samples: Sequence[WindowSample],
                 window_size: int) -> None:
        self.data = data
        self.samples = list(samples)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        flux = self.data["flux"][sample.spec_idx]
        window = extract_window(flux, sample.center_idx, self.window_size)[None, :]
        z_qso = np.float32(self.data["z_qso"][sample.spec_idx])
        center_wave = np.float32(self.data["wave"][sample.center_idx])
        aux = np.array([center_wave / 8000.0, z_qso / 4.0], dtype=np.float32)
        return {
            "window": torch.from_numpy(window),
            "aux": torch.from_numpy(aux),
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "offset": torch.tensor(sample.offset_norm, dtype=torch.float32),
            "logn": torch.tensor(sample.logn_norm, dtype=torch.float32),
        }


class WindowCnn(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.aux = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 16, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.cls_head = nn.Linear(hidden_dim, 1)
        self.offset_head = nn.Linear(hidden_dim, 1)
        self.logn_head = nn.Linear(hidden_dim, 1)

    def forward(self, window: torch.Tensor, aux: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(window).squeeze(-1)
        aux_feat = self.aux(aux)
        x = self.head(torch.cat([feat, aux_feat], dim=1))
        return {
            "logit": self.cls_head(x).squeeze(1),
            "offset": torch.tanh(self.offset_head(x).squeeze(1)),
            "logn_norm": torch.sigmoid(self.logn_head(x).squeeze(1)),
        }


def compute_window_loss(outputs: Dict[str, torch.Tensor],
                        batch: Dict[str, torch.Tensor],
                        pos_weight: float = 1.0) -> Dict[str, torch.Tensor]:
    pw = torch.tensor([pos_weight], device=outputs["logit"].device, dtype=outputs["logit"].dtype)
    cls = nn.functional.binary_cross_entropy_with_logits(
        outputs["logit"], batch["label"], pos_weight=pw
    )
    pos_mask = batch["label"]
    off_raw = nn.functional.smooth_l1_loss(outputs["offset"], batch["offset"], reduction="none")
    logn_raw = nn.functional.smooth_l1_loss(outputs["logn_norm"], batch["logn"], reduction="none")
    off = (off_raw * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
    logn = (logn_raw * pos_mask).sum() / pos_mask.sum().clamp_min(1.0)
    loss = cls + 0.5 * off + 0.5 * logn
    return {"loss": loss, "cls": cls.detach(), "off": off.detach(), "logn": logn.detach()}


def candidate_rank(cand: Dict[str, float], rank_by: str = "confidence") -> float:
    if rank_by == "confidence":
        return float(cand["confidence"])
    if rank_by == "logn":
        return float(cand["log_nhi"])
    if rank_by == "conf_logn":
        return float(cand["confidence"]) * max(float(cand["log_nhi"]) - 20.3, 0.0)
    if rank_by == "support":
        return float(cand.get("support", 1.0))
    if rank_by == "conf_support":
        return float(cand["confidence"]) * float(cand.get("support", 1.0))
    if rank_by == "mean_conf":
        return float(cand.get("mean_conf", cand["confidence"]))
    if rank_by == "cluster_score":
        return (
            float(cand["confidence"])
            * np.sqrt(max(float(cand.get("support", 1.0)), 1.0))
            * max(float(cand["log_nhi"]) - 20.3, 0.0)
        )
    raise ValueError(f"Unknown rank_by={rank_by!r}")


def merge_candidates(candidates: List[Dict[str, float]],
                     min_separation_pix: int = 80,
                     rank_by: str = "confidence") -> List[Dict[str, float]]:
    if not candidates:
        return []

    remaining = list(candidates)
    clusters: List[List[Dict[str, float]]] = []
    # Greedy peak clustering avoids chaining many low-threshold windows into a
    # spectrum-scale cluster. Each cluster is local to its selected peak.
    while remaining:
        remaining.sort(key=lambda x: candidate_rank(x, rank_by), reverse=True)
        peak = remaining[0]
        peak_center = float(peak["center_pix"])
        cluster = [
            cand for cand in remaining
            if abs(float(cand["center_pix"]) - peak_center) <= min_separation_pix
        ]
        clusters.append(cluster)
        cluster_ids = {id(cand) for cand in cluster}
        remaining = [cand for cand in remaining if id(cand) not in cluster_ids]

    merged: List[Dict[str, float]] = []
    for cluster in clusters:
        conf = np.asarray([float(c["confidence"]) for c in cluster], dtype=np.float64)
        weights = np.clip(conf, 1e-6, None)
        centers = np.asarray([float(c["center_pix"]) for c in cluster], dtype=np.float64)
        z_dlas = np.asarray([float(c["z_dla"]) for c in cluster], dtype=np.float64)
        logns = np.asarray([float(c["log_nhi"]) for c in cluster], dtype=np.float64)
        peak_idx = int(np.argmax(conf))
        peak = cluster[peak_idx]
        merged.append({
            "center_pix": float(np.average(centers, weights=weights)),
            "z_dla": float(np.average(z_dlas, weights=weights)),
            "log_nhi": float(np.average(logns, weights=weights)),
            "confidence": float(conf[peak_idx]),
            "peak_center_pix": float(peak["center_pix"]),
            "peak_z_dla": float(peak["z_dla"]),
            "peak_log_nhi": float(peak["log_nhi"]),
            "support": float(len(cluster)),
            "mean_conf": float(np.mean(conf)),
            "sum_conf": float(np.sum(conf)),
            "width_pix": float(np.max(centers) - np.min(centers)) if len(cluster) > 1 else 0.0,
        })

    merged = sorted(merged, key=lambda x: candidate_rank(x, rank_by), reverse=True)
    return merged
