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
Z_MIN = 1.6
LOGNHI_MIN = 19.5
LOGNHI_MAX = 22.5


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


def normalize_dla_params(z: np.ndarray, logn: np.ndarray, z_qso: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z_max = np.maximum(z_qso - 0.05, Z_MIN + 1e-3)
    z_norm = (z - Z_MIN) / np.maximum(z_max - Z_MIN, 1e-3)
    logn_norm = (logn - LOGNHI_MIN) / (LOGNHI_MAX - LOGNHI_MIN)
    return z_norm.astype(np.float32), logn_norm.astype(np.float32)


def denormalize_dla_params(params: np.ndarray, z_qso: np.ndarray) -> np.ndarray:
    params = np.clip(params, 0.0, 1.0)
    z_max = np.maximum(z_qso - 0.05, Z_MIN + 1e-3)
    z_scale = np.maximum(z_max - Z_MIN, 1e-3)
    out = np.empty_like(params, dtype=np.float32)
    out[:, 0] = Z_MIN + params[:, 0] * z_scale
    out[:, 1] = LOGNHI_MIN + params[:, 1] * (LOGNHI_MAX - LOGNHI_MIN)
    out[:, 2] = Z_MIN + params[:, 2] * z_scale
    out[:, 3] = LOGNHI_MIN + params[:, 3] * (LOGNHI_MAX - LOGNHI_MIN)
    return out


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
            self.two_dla = (self.n_dla == 2).astype(np.float32)
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
        z_qso = np.float32(self.aux[idx, 0])

        if not self.training:
            return {
                "flux": torch.from_numpy(flux),
                "aux": torch.from_numpy(aux),
                "z_qso": torch.tensor(z_qso, dtype=torch.float32),
            }

        z_qso = float(self.aux[idx, 0])
        z1 = np.nan_to_num(self.z1[idx], nan=Z_MIN)
        z2 = np.nan_to_num(self.z2[idx], nan=Z_MIN)
        n1 = np.nan_to_num(self.logn1[idx], nan=LOGNHI_MIN)
        n2 = np.nan_to_num(self.logn2[idx], nan=LOGNHI_MIN)
        z_norm, logn_norm = normalize_dla_params(
            np.array([z1, z2], dtype=np.float32),
            np.array([n1, n2], dtype=np.float32),
            np.array([z_qso, z_qso], dtype=np.float32),
        )
        target = np.array([z_norm[0], logn_norm[0], z_norm[1], logn_norm[1]], dtype=np.float32)
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
            "two_dla": torch.tensor(self.two_dla[idx], dtype=torch.float32),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "z_qso": torch.tensor(z_qso, dtype=torch.float32),
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
        self.two_dla_head = nn.Linear(hidden_dim, 1)
        self.param_head = nn.Linear(hidden_dim, 4)

    def forward(self, flux: torch.Tensor, aux: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(flux)
        x = self.encoder(x)
        x = self.pool(x).squeeze(-1)
        aux_feat = self.aux_mlp(aux)
        feat = self.head(torch.cat([x, aux_feat], dim=1))
        return {
            "has_dla_logit": self.has_dla_head(feat).squeeze(1),
            "two_dla_logit": self.two_dla_head(feat).squeeze(1),
            "params_norm": torch.sigmoid(self.param_head(feat)),
        }


def focal_bce_with_logits(logits: torch.Tensor,
                          targets: torch.Tensor,
                          alpha: float = 0.25,
                          gamma: float = 2.0,
                          pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    return (alpha_t * modulating * bce).mean()


def compute_losses(outputs: Dict[str, torch.Tensor],
                   batch: Dict[str, torch.Tensor],
                   loss_config: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    if loss_config is None:
        loss_config = {
            "has_dla": 1.0,
            "two_dla": 0.35,
            "params": 0.8,
            "pos_weight": 1.0,
            "focal_alpha": 0.35,
            "focal_gamma": 2.0,
        }

    pos_weight = torch.tensor(
        [loss_config["pos_weight"]],
        device=outputs["has_dla_logit"].device,
        dtype=outputs["has_dla_logit"].dtype,
    )
    has_dla_loss = focal_bce_with_logits(
        outputs["has_dla_logit"],
        batch["has_dla"],
        alpha=loss_config["focal_alpha"],
        gamma=loss_config["focal_gamma"],
        pos_weight=pos_weight,
    )
    two_raw = nn.functional.binary_cross_entropy_with_logits(
        outputs["two_dla_logit"], batch["two_dla"], reduction="none"
    )
    two_mask = batch["has_dla"]
    two_dla_loss = (two_raw * two_mask).sum() / two_mask.sum().clamp_min(1.0)
    raw_reg = nn.functional.smooth_l1_loss(
        outputs["params_norm"], batch["target"], reduction="none"
    )
    mask = batch["mask"]
    reg = (raw_reg * mask).sum() / mask.sum().clamp_min(1.0)
    total = (
        loss_config["has_dla"] * has_dla_loss
        + loss_config["two_dla"] * two_dla_loss
        + loss_config["params"] * reg
    )
    return {
        "loss": total,
        "bce": has_dla_loss.detach(),
        "ce": two_dla_loss.detach(),
        "reg": reg.detach(),
    }


def decode_count_predictions(has_probs: np.ndarray,
                             two_probs: np.ndarray,
                             threshold_has: float = 0.5,
                             threshold_two: float = 0.5) -> np.ndarray:
    counts = np.zeros_like(has_probs, dtype=np.int64)
    pos = has_probs >= threshold_has
    counts[pos] = 1
    counts[pos & (two_probs >= threshold_two)] = 2
    return counts


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
