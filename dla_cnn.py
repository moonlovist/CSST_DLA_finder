#!/usr/bin/env python3
"""Shared utilities for CSST DLA CNN training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from astropy.io import fits
from torch.utils.data import Dataset


EPS = 1e-6


def estimate_band_snrs(wave: np.ndarray, flux: np.ndarray) -> np.ndarray:
    bands = [
        wave < 4100.0,
        (wave >= 4100.0) & (wave < 6200.0),
        wave >= 6200.0,
    ]
    snrs = []
    for mask in bands:
        band_flux = flux[:, mask]
        median = np.median(band_flux, axis=1)
        mad = np.median(np.abs(band_flux - median[:, None]), axis=1)
        sigma = 1.4826 * mad
        sigma = np.where(np.isfinite(sigma) & (sigma > EPS), sigma, 1.0)
        snr = np.abs(median) / sigma
        snrs.append(snr.astype(np.float32))
    return np.stack(snrs, axis=1)


def _safe_std(x: np.ndarray) -> np.ndarray:
    std = x.std(axis=0).astype(np.float32)
    std[std < EPS] = 1.0
    return std


def _sort_dla_targets(z1: np.ndarray,
                      n1: np.ndarray,
                      z2: np.ndarray,
                      n2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z1 = z1.copy()
    n1 = n1.copy()
    z2 = z2.copy()
    n2 = n2.copy()
    swap = np.isfinite(z1) & np.isfinite(z2) & (z2 < z1)
    if np.any(swap):
        z1_swap = z1[swap].copy()
        n1_swap = n1[swap].copy()
        z1[swap] = z2[swap]
        n1[swap] = n2[swap]
        z2[swap] = z1_swap
        n2[swap] = n1_swap
    return z1, n1, z2, n2


def load_fits_dataset(path: str, expect_labels: bool = True) -> Dict[str, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float32)
        flux = np.asarray(hdul["FLUX"].data, dtype=np.float32)

        result: Dict[str, np.ndarray] = {
            "wave": wave,
            "flux": flux,
        }

        if expect_labels:
            labels = hdul["LABELS"].data
            aux = np.stack([
                np.asarray(labels["Z_QSO"], dtype=np.float32),
                np.asarray(labels["SNR_GU"], dtype=np.float32),
                np.asarray(labels["SNR_GV"], dtype=np.float32),
                np.asarray(labels["SNR_GI"], dtype=np.float32),
            ], axis=1)
            has_dla = np.asarray(labels["HAS_DLA"], dtype=np.int64)
            n_dla = np.asarray(labels["N_DLA"], dtype=np.int64)
            z1 = np.asarray(labels["Z_DLA1"], dtype=np.float32)
            logn1 = np.asarray(labels["LOGNHI1"], dtype=np.float32)
            z2 = np.asarray(labels["Z_DLA2"], dtype=np.float32)
            logn2 = np.asarray(labels["LOGNHI2"], dtype=np.float32)
            z1, logn1, z2, logn2 = _sort_dla_targets(z1, logn1, z2, logn2)
            result.update({
                "aux": aux,
                "has_dla": has_dla,
                "n_dla": n_dla,
                "z1": z1,
                "logn1": logn1,
                "z2": z2,
                "logn2": logn2,
            })

        else:
            if "LABELS" in hdul:
                meta = hdul["LABELS"].data
                z_qso = np.asarray(meta["Z_QSO"], dtype=np.float32)
                if all(name in meta.names for name in ["SNR_GU", "SNR_GV", "SNR_GI"]):
                    snr = np.stack([
                        np.asarray(meta["SNR_GU"], dtype=np.float32),
                        np.asarray(meta["SNR_GV"], dtype=np.float32),
                        np.asarray(meta["SNR_GI"], dtype=np.float32),
                    ], axis=1)
                else:
                    snr = estimate_band_snrs(wave, flux)
                targetid = np.arange(flux.shape[0], dtype=np.int64)
            else:
                meta = hdul["META"].data
                z_qso = np.asarray(meta["Z_QSO"], dtype=np.float32)
                if "TARGETID" in meta.names:
                    targetid = np.asarray(meta["TARGETID"], dtype=np.int64)
                else:
                    targetid = np.arange(flux.shape[0], dtype=np.int64)
                snr = estimate_band_snrs(wave, flux)
            aux = np.concatenate([z_qso[:, None], snr], axis=1).astype(np.float32)
            result.update({
                "aux": aux,
                "targetid": targetid,
                "z_qso": z_qso,
            })

        return result


def compute_flux_scale(flux: np.ndarray) -> np.ndarray:
    scale = np.percentile(np.abs(flux), 95, axis=1).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    return scale


@dataclass
class NormalizationStats:
    aux_mean: np.ndarray
    aux_std: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "aux_mean": self.aux_mean.astype(np.float32),
            "aux_std": self.aux_std.astype(np.float32),
        }


def build_normalization_stats(aux: np.ndarray, indices: np.ndarray) -> NormalizationStats:
    sub = aux[indices]
    return NormalizationStats(
        aux_mean=sub.mean(axis=0).astype(np.float32),
        aux_std=_safe_std(sub),
    )


class DlaSpectraDataset(Dataset):
    def __init__(self,
                 data: Dict[str, np.ndarray],
                 indices: np.ndarray,
                 norm_stats: NormalizationStats,
                 training: bool = True) -> None:
        self.wave = data["wave"]
        self.flux = data["flux"][indices]
        self.aux = data["aux"][indices]
        self.indices = indices
        self.training = training
        self.norm_stats = norm_stats
        self.scale = compute_flux_scale(self.flux)
        self.aux_norm = ((self.aux - norm_stats.aux_mean) / norm_stats.aux_std).astype(np.float32)

        if training:
            self.has_dla = data["has_dla"][indices].astype(np.float32)
            self.n_dla = data["n_dla"][indices].astype(np.int64)
            self.z1 = data["z1"][indices].astype(np.float32)
            self.logn1 = data["logn1"][indices].astype(np.float32)
            self.z2 = data["z2"][indices].astype(np.float32)
            self.logn2 = data["logn2"][indices].astype(np.float32)

    def __len__(self) -> int:
        return self.flux.shape[0]

    def __getitem__(self, idx: int):
        flux = self.flux[idx] / self.scale[idx]
        flux = np.clip(flux, -5.0, 5.0).astype(np.float32)
        flux = flux[None, :]
        aux = self.aux_norm[idx]

        if not self.training:
            return {
                "flux": torch.from_numpy(flux),
                "aux": torch.from_numpy(aux),
            }

        target = np.array([
            np.nan_to_num(self.z1[idx], nan=0.0),
            np.nan_to_num(self.logn1[idx], nan=0.0),
            np.nan_to_num(self.z2[idx], nan=0.0),
            np.nan_to_num(self.logn2[idx], nan=0.0),
        ], dtype=np.float32)
        mask = np.array([
            float(np.isfinite(self.z1[idx])),
            float(np.isfinite(self.logn1[idx])),
            float(np.isfinite(self.z2[idx])),
            float(np.isfinite(self.logn2[idx])),
        ], dtype=np.float32)

        return {
            "flux": torch.from_numpy(flux),
            "aux": torch.from_numpy(aux),
            "has_dla": torch.tensor(self.has_dla[idx], dtype=torch.float32),
            "n_dla": torch.tensor(self.n_dla[idx], dtype=torch.long),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
        }


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=7, stride=stride, padding=3, bias=False),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(c_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)


class DlaCnnModel(nn.Module):
    def __init__(self, aux_dim: int = 4, hidden_dim: int = 256) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.encoder = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 192, stride=2),
            ConvBlock(192, 256, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 32, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.has_dla_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Linear(hidden_dim, 3)
        self.param_head = nn.Linear(hidden_dim, 4)

    def forward(self, flux: torch.Tensor, aux: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(flux)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        aux_feat = self.aux_mlp(aux)
        feat = self.head(torch.cat([x, aux_feat], dim=1))
        return {
            "has_dla_logit": self.has_dla_head(feat).squeeze(1),
            "count_logit": self.count_head(feat),
            "params": self.param_head(feat),
        }


def compute_losses(outputs: Dict[str, torch.Tensor],
                   batch: Dict[str, torch.Tensor],
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = {
            "has_dla": 1.0,
            "count": 0.5,
            "params": 1.0,
        }

    bce = nn.functional.binary_cross_entropy_with_logits(
        outputs["has_dla_logit"], batch["has_dla"]
    )
    ce = nn.functional.cross_entropy(outputs["count_logit"], batch["n_dla"])
    raw_reg = nn.functional.smooth_l1_loss(
        outputs["params"], batch["target"], reduction="none"
    )
    mask = batch["mask"]
    reg = (raw_reg * mask).sum() / mask.sum().clamp_min(1.0)
    total = weights["has_dla"] * bce + weights["count"] * ce + weights["params"] * reg
    return {
        "loss": total,
        "bce": bce.detach(),
        "ce": ce.detach(),
        "reg": reg.detach(),
    }


def classification_metrics(probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    preds = (probs >= 0.5).astype(np.int64)
    targets = targets.astype(np.int64)
    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    acc = float((tp + tn) / max(len(targets), 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, EPS))
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def rmse_masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    denom = float(mask.sum())
    if denom < 1:
        return math.nan
    return float(np.sqrt((((pred - target) ** 2) * mask).sum() / denom))
