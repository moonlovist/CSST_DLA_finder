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
                         jitter_pix: int = 6,
                         seed: int = 20251031) -> List[WindowSample]:
    wave = data["wave"]
    rng = np.random.default_rng(seed)
    half = window_size // 2
    low = half
    high = len(wave) - half - 1
    samples: List[WindowSample] = []

    for spec_idx in indices:
        centers = dla_centers_pix(wave, data["z1"][spec_idx], data["z2"][spec_idx])
        for j, c in enumerate(centers):
            for _ in range(3):
                jitter = int(rng.integers(-jitter_pix, jitter_pix + 1))
                anchor = int(np.clip(c + jitter, low, high))
                offset = (c - anchor) / max(jitter_pix, 1)
                logn = data["logn1"][spec_idx] if j == 0 else data["logn2"][spec_idx]
                logn_norm = (float(logn) - LOGNHI_MIN) / (LOGNHI_MAX - LOGNHI_MIN)
                samples.append(WindowSample(spec_idx, anchor, 1, float(offset), float(logn_norm)))

        forbidden = np.zeros(len(wave), dtype=bool)
        for c in centers:
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


def merge_candidates(candidates: List[Dict[str, float]], min_separation_pix: int = 20) -> List[Dict[str, float]]:
    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    kept: List[Dict[str, float]] = []
    for cand in candidates:
        if all(abs(cand["center_pix"] - prev["center_pix"]) > min_separation_pix for prev in kept):
            kept.append(cand)
    kept.sort(key=lambda x: x["center_pix"])
    return kept
