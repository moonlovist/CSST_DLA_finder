#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_csst_dla_training.py
─────────────────────────────────────────────────────────────────────────────
批量生成 CSST-like QSO 模拟光谱，用于 DLA finder 训练。

所有光谱处理逻辑严格遵循原始 DLA_spectrum.ipynb：
  - DLA Voigt 轮廓：voigt_tau / dla_spec / insert_dlas 原样保留
  - 分辨率退化：sigma_pix = (8.0 * R_GRID) / (2.355 * lam_eff)
                 其中 R_GRID=3000, lam_eff=3600（固定常数，与原始一致）
  - 噪声模型：
      scale   = percentile(|degraded|, 95)
      f_norm  = degraded / scale
      GU band (λ < lambda_cut=4100 Å):
          sigma_noise = sqrt(min_sigma^2 + (|f_norm| / snr_gu)^2)
          noisy = f_norm + Normal(0, sigma_noise),  clip to >=0
      GV band (λ >= lambda_cut=4100 Å):
          sigma_noise = sqrt(min_sigma^2 + (|f_norm| / snr_gv)^2)
          noisy = f_norm + Normal(0, sigma_noise),  clip to >=0
      flux_out = noisy * scale
  - 波长网格：sqbase.fixed_R_dispersion(2000, 8000, 2000)

批量配置
────────
  z_qso      : 均匀分布 [1.6, 4.0]
  DLA 比例   : 50% QSO 无 DLA，50% 含 DLA
  DLA 数量   : DLA-QSO 中 75% 含 1 个，25% 含 2 个
  z_dla      : 均匀分布 [1.6, z_qso - 0.05]
  log N_HI   : [19.5, 22.5]，f(N) ∝ N^{-1.5} 幂律
  snr_gu     : 每条谱独立从 Uniform(0.6, 1.0) 采样
  snr_gv     : 固定 2.2
  min_sigma  : 0.02

输出：fits.gz
────
  PRIMARY    : 全局元数据
  WAVELENGTH : 1D float32 [N_pix]，单位 Å
  FLUX       : 2D float32 [N_spec x N_pix]，含噪含 DLA
  FLUX_CLEAN : 2D float32 [N_spec x N_pix]，无噪（降分辨率+DLA）
  LABELS     : BinTable — Z_QSO, HAS_DLA, N_DLA,
                           Z_DLA1, LOGNHI1, Z_DLA2, LOGNHI2, SNR_GU

用法
────
  python generate_csst_dla_training.py \
      --n_spectra 10000 \
      --output ./csst_training/train.fits.gz \
      --batch_size 200 \
      --seed 20251031 \
      --simqso_path /global/cfs/cdirs/desi/users/tingtan/CSST_DLA/simqso/desisim/
"""

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from astropy.io import fits
from astropy.cosmology import Planck13
from astropy import constants as const
from scipy.ndimage import gaussian_filter1d
from scipy.special import wofz

# ══════════════════════════════════════════════════════════════════════════════
# 0.  命令行参数
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch CSST QSO+DLA spectrum generator for DLA finder training")
    p.add_argument("--n_spectra",   type=int,   default=10_000)
    p.add_argument("--batch_size",  type=int,   default=200)
    p.add_argument("--output",      type=str,
                   default="./csst_dla_training/training_set.fits.gz")
    p.add_argument("--seed",        type=int,   default=20251031)
    p.add_argument("--simqso_path", type=str,
                   default=os.environ.get(
                       "SIMQSO_PATH",
                       "/global/cfs/cdirs/desi/users/tingtan/CSST_DLA/simqso/desisim/"))
    p.add_argument("--no_clean",    action="store_true",
                   help="Skip FLUX_CLEAN extension to save disk space")
    p.add_argument("--n_workers",   type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                   help="Number of CPU threads for per-spectrum processing inside each batch")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DLA Voigt 轮廓 —— 逐字照抄原始 DLA_spectrum.ipynb
# ══════════════════════════════════════════════════════════════════════════════

c_cgs = const.c.to('cm/s').value   # speed of light [cm/s]


def voigt_wofz(vin, a):
    """Voigt profile via Faddeeva function (real part of wofz)."""
    return wofz(vin + 1j * a).real


def voigt_tau(wave_cm, par):
    """
    Optical depth tau at wavelengths (in cm).

    par list:
      par[0] = log10 N_HI [cm^-2]
      par[1] = z_abs
      par[2] = b [cm/s]
      par[3] = line rest wavelength [cm]  (Lya)
      par[4] = oscillator strength f
      par[5] = gamma [s^-1]
    """
    cold   = 10.0 ** par[0]
    zp1    = par[1] + 1.0
    nujk   = c_cgs / par[3]
    dnu    = par[2] / par[3]
    avoigt = par[5] / (4 * np.pi * dnu)
    uvoigt = ((c_cgs / (wave_cm / zp1)) - nujk) / dnu
    cne    = 0.014971475 * cold * par[4]
    tau    = cne * voigt_wofz(uvoigt, avoigt) / dnu
    return tau


def dla_spec(wave_ang, dlas):
    """
    Build multiplicative absorption model from a list of DLAs.
      wave_ang : observed wavelengths in Angstrom
      dlas     : list of dicts with keys {'z', 'N', 'dlaid'}
                 where N is log10(N_HI/cm^2)
    Returns: flux transmission exp(-tau)
    """
    flya      = 0.4164
    gamma_lya = 6.265e8
    lyacm     = 1215.6700 / 1e8
    wavecm    = wave_ang / 1e8

    tau = np.zeros(wave_ang.size, dtype=float)
    for dla in dlas:
        par = [dla['N'],
               dla['z'],
               30 * 1e5,   # b = 30 km/s -> 3e6 cm/s
               lyacm,
               flya,
               gamma_lya]
        tau += voigt_tau(wavecm, par)
    return np.exp(-tau)


def insert_dlas(wave_ang, dla_list):
    """
    Convenience wrapper: insert one or more DLAs.
      dla_list : list of (z_dla, logNHI) tuples
    Returns transmission array (same shape as wave_ang).
    """
    dlas = [dict(z=z, N=N, dlaid=i) for i, (z, N) in enumerate(dla_list)]
    return dla_spec(wave_ang, dlas)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  N_HI 幂律采样  f(N) ∝ N^{-beta}
# ══════════════════════════════════════════════════════════════════════════════

def sample_logNHI(rng, n=1, beta=1.5, logN_min=19.5, logN_max=22.5):
    """
    从截断幂律 f(N) ∝ N^{-beta} 采样 log10(N_HI)，使用逆 CDF 方法。
    beta=1.5 -> 大 N_HI 系统显著更少。
    """
    alpha = 1.0 - beta
    N_min = 10.0 ** logN_min
    N_max = 10.0 ** logN_max
    u = rng.uniform(0.0, 1.0, size=n)
    N = (N_min**alpha + u * (N_max**alpha - N_min**alpha)) ** (1.0 / alpha)
    return np.log10(N).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  filter_like_pipeline —— 严格照抄原始 DLA_spectrum.ipynb
#
#     唯一扩展：原始只对 GU band (wave < lambda_cut) 加噪。
#     此处额外对 GV band (wave >= lambda_cut) 也加噪（snr_gv=2.2），
#     噪声公式与 GU 完全一致，仅 SNR 参数不同。
# ══════════════════════════════════════════════════════════════════════════════

# 固定常数（与原始 notebook 完全一致）
_R_GRID    = 3000.0
_LAM_EFF   = 3600.0
_SIGMA_PIX = (8.0 * _R_GRID) / (2.355 * _LAM_EFF)   # ~2.828 pix，固定值


def filter_like_pipeline(wave, flux,
                         lcut_gv=4100.0, lcut_gi=6200.0,
                         min_sigma=0.02,
                         snr_gu=1.0, snr_gv=2.0, snr_gi=3.0,
                         rng=None,
                         noise_norms=None):
    """
    Step 1 — 分辨率退化（原始代码逐字）：
        sigma_pix = (8.0 * 3000.0) / (2.355 * 3600.0)  ~= 2.828 pix
        degraded  = gaussian_filter1d(flux, sigma_pix)

    Step 2 — 归一化（原始）：
        scale  = percentile(|degraded|, 95)
        f_norm = degraded / scale

    Step 3 — 三波段加噪（sigma_noise = sqrt(min_sigma^2 + (|f_norm|/SNR)^2)）：
        GU : wave < lcut_gv               SNR = snr_gu
        GV : lcut_gv <= wave < lcut_gi    SNR = snr_gv
        GI : wave >= lcut_gi              SNR = snr_gi

    Step 4 — 反归一化：flux_out = noisy * scale
    """
    if rng is None:
        rng = np.random.default_rng(20251031)

    # ── Step 1 ───────────────────────────────────────────────────────────────
    flux = np.nan_to_num(np.asarray(flux, dtype=np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)
    degraded = gaussian_filter1d(flux, sigma=_SIGMA_PIX)

    # ── Step 2 ───────────────────────────────────────────────────────────────
    scale = np.percentile(np.abs(degraded), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(np.mean(np.abs(degraded)), 1.0)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    f_norm = degraded / scale

    # ── Step 3 ───────────────────────────────────────────────────────────────
    noisy = f_norm.copy()

    for iband, (mask, snr) in enumerate([
        (wave < lcut_gv,                          snr_gu),
        ((wave >= lcut_gv) & (wave < lcut_gi),    snr_gv),
        (wave >= lcut_gi,                          snr_gi),
    ]):
        if not np.any(mask):
            continue
        sig = np.sqrt(min_sigma**2 + (np.abs(f_norm[mask]) / snr)**2)
        if noise_norms is None:
            noisy[mask] = f_norm[mask] + rng.normal(0.0, sig, size=mask.sum())
        else:
            noisy[mask] = f_norm[mask] + np.asarray(noise_norms[iband], dtype=np.float64) * sig

    # ── Step 4 ───────────────────────────────────────────────────────────────
    return (noisy * scale).astype(np.float32)


def process_one_spectrum(payload):
    """Process one spectrum after simqso generation.

    The random numbers are pre-drawn by the parent process in the original serial
    order, so threaded execution does not change the generated data for a fixed seed.
    """
    (gi, spec, wave, has_dla_i, n_dla_i, z_dla1_i, logNHI1_i, z_dla2_i, logNHI2_i,
     ratio_gv_draw, ratio_gi_draw, noise_norms, no_clean, lcut_gv, lcut_gi,
     min_sigma, f0, snr_gv_ratio, snr_gi_ratio, n_pix) = payload

    gu_mask = wave < lcut_gv
    f_gu_mean = float(np.mean(spec[gu_mask]))
    if f_gu_mean > 0:
        log_snr14 = -0.261 + 0.469 * np.log10(f_gu_mean / f0) + 0.4
        snr14 = 10.0 ** log_snr14
        snr_gu = snr14 * np.sqrt(8.0 / 14.0)
    else:
        snr_gu = 0.8
    snr_gu = max(snr_gu, 0.05)

    ratio_gv = snr_gv_ratio * (1.0 + ratio_gv_draw)
    ratio_gi = snr_gi_ratio * (1.0 + ratio_gi_draw)
    ratio_gv = max(ratio_gv, 1.5)
    ratio_gi = max(ratio_gi, 1.5)
    snr_gv = snr_gu * ratio_gv
    snr_gi = snr_gv * ratio_gi

    if has_dla_i:
        dla_list = [(float(z_dla1_i), float(logNHI1_i))]
        if n_dla_i == 2:
            dla_list.append((float(z_dla2_i), float(logNHI2_i)))
        trans = insert_dlas(wave, dla_list)
    else:
        trans = np.ones(n_pix, dtype=np.float64)

    spec_dla = spec * trans
    clean = None
    if not no_clean:
        spec_dla_clean = np.nan_to_num(
            spec_dla.astype(np.float64),
            nan=0.0, posinf=0.0, neginf=0.0)
        clean = gaussian_filter1d(spec_dla_clean, sigma=_SIGMA_PIX).astype(np.float32)

    noisy = filter_like_pipeline(
        wave, spec_dla,
        lcut_gv=lcut_gv, lcut_gi=lcut_gi,
        min_sigma=min_sigma,
        snr_gu=snr_gu, snr_gv=snr_gv, snr_gi=snr_gi,
        noise_norms=noise_norms,
    )
    return gi, noisy, clean, np.float32(snr_gu), np.float32(snr_gv), np.float32(snr_gi)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  simqso 批量 QSO 生成
# ══════════════════════════════════════════════════════════════════════════════

def generate_qso_batch(wave, z_qso_list, batch_idx=0, master_seed=20251031):
    """
    调用 simqso 批量生成原始 QSO 光谱（含 Lya forest，无 DLA，无噪）。

    波长网格：fixed_R_dispersion(2000, 8000, 2000)，与原始 notebook 一致。
    每批 seed 通过 batch_idx 偏移，保证批间光谱独立。
    """
    from simqso.sqgrids import generateQlfPoints
    from simqso import sqbase                             # noqa
    from simqso.sqrun import buildSpectraBulk
    from simqso.sqmodels import BOSS_DR9_PLEpivot, get_BossDr9_model_vars

    seed_off = batch_idx * 3
    kcorr = sqbase.ContinuumKCorr('DECam-r', 1450, effWaveBand='SDSS-r')
    qsos = generateQlfPoints(
        BOSS_DR9_PLEpivot(cosmo=Planck13),
        (19, 22), (1.6, 4.0),
        kcorr=kcorr,
        zin=list(z_qso_list),
        qlfseed=master_seed + seed_off,
        gridseed=master_seed + seed_off + 1,
    )
    sed_vars = get_BossDr9_model_vars(
        qsos, wave, 0,
        forestseed=master_seed + seed_off + 2,
        verbose=0,
    )
    qsos.addVars(sed_vars)
    qsos.loadPhotoMap([('DECam', 'DECaLS'), ('WISE', 'AllWISE')])
    # saveSpectra=True 是关键：False 时 simqso 不填充 spectra 列表，
    # 与原始 notebook (saveSpectra=True) 保持一致。
    _, spectra = buildSpectraBulk(wave, qsos, saveSpectra=True,
                                  maxIter=3, verbose=0)

    # simqso 可能返回 masked array 或带 .flux 属性的 Spectrum 对象；
    # 统一提取为干净的 float64 ndarray，masked 值填 0。
    result = []
    for s in spectra:
        if hasattr(s, 'flux'):
            arr = np.asarray(s.flux, dtype=np.float64)
        else:
            arr = np.asarray(s, dtype=np.float64)
        if np.ma.is_masked(arr):
            arr = np.ma.filled(arr, fill_value=0.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        result.append(arr)
    return np.array(result, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    sys.path.insert(0, args.simqso_path)
    print(f"Loading simqso from {args.simqso_path} ...", flush=True)
    from simqso import sqbase   # verify import early
    print("simqso import OK", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # 波长网格：与原始 notebook 完全一致
    wave  = sqbase.fixed_R_dispersion(2000, 8000, 2000)
    N_PIX = len(wave)

    # 超参数
    LCUT_GV    = 4100.0    # GU/GV 分界 [A]
    LCUT_GI    = 6200.0    # GV/GI 分界 [A]
    MIN_SIGMA  = 0.02
    NHI_BETA   = 1.5
    Z_QSO_MIN  = 1.6
    Z_QSO_MAX  = 4.0
    DLA_FRAC   = 0.5
    TWO_FRAC   = 0.25
    N          = args.n_spectra
    BS         = args.batch_size
    # SNR 关联参数（由 GU flux 驱动）
    F0            = 1e-17        # 参考 flux [erg/s/cm^2]，公式中 f0
    SNR_GV_RATIO  = 3.0          # SNR_GV ≈ SNR_GU × ratio（含散射）
    SNR_GI_RATIO  = 1.5          # SNR_GI ≈ SNR_GV × ratio（含散射）
    SNR_SCATTER   = 0.25         # ratio 的相对散射 (12%)，给 ~0.1-0.2 dex 变化

    print("=" * 52)
    print("  CSST DLA Training Set Generator")
    print("=" * 52)
    print(f"  N spectra  : {N}")
    print(f"  Batch size : {BS}")
    print(f"  Wave grid  : 2000-8000 A  ({N_PIX} pix)")
    print(f"  sigma_pix  : {_SIGMA_PIX:.4f} pix  (fixed, R_GRID=3000 lam_eff=3600)")
    print(f"  Bands      : GU<{LCUT_GV:.0f}  GV {LCUT_GV:.0f}-{LCUT_GI:.0f}  GI>{LCUT_GI:.0f} A")
    print(f"  SNR_GU     : from flux formula (Eq.9, scaled to 8A bins)")
    print(f"  SNR_GV     : SNR_GU x {SNR_GV_RATIO} (+/-{SNR_SCATTER*100:.0f}%)")
    print(f"  SNR_GI     : SNR_GV x {SNR_GI_RATIO} (+/-{SNR_SCATTER*100:.0f}%)")
    print(f"  Workers    : {args.n_workers}")
    print(f"  Output     : {args.output}")
    print("=" * 52 + "\n")

    # 预分配
    flux_noisy = np.zeros((N, N_PIX), dtype=np.float32)
    flux_clean = np.zeros((N, N_PIX), dtype=np.float32) if not args.no_clean else None

    # 预采样所有标签
    z_qso_arr  = rng.uniform(Z_QSO_MIN, Z_QSO_MAX, N).astype(np.float32)
    has_dla    = np.zeros(N, dtype=np.int16)
    n_dla      = np.zeros(N, dtype=np.int16)
    z_dla1     = np.full(N, np.nan, dtype=np.float32)
    logNHI1    = np.full(N, np.nan, dtype=np.float32)
    z_dla2     = np.full(N, np.nan, dtype=np.float32)
    logNHI2    = np.full(N, np.nan, dtype=np.float32)

    dla_draw = rng.uniform(0, 1, N)
    two_draw = rng.uniform(0, 1, N)

    for i in range(N):
        if dla_draw[i] >= DLA_FRAC:
            continue
        z_max = float(z_qso_arr[i]) - 0.05
        z_min = Z_QSO_MIN
        if z_max <= z_min:
            continue

        has_dla[i] = 1
        z1 = float(rng.uniform(z_min, z_max))
        N1 = float(sample_logNHI(rng, n=1, beta=NHI_BETA)[0])
        z_dla1[i]  = z1
        logNHI1[i] = N1

        if two_draw[i] < TWO_FRAC:
            n_dla[i] = 2
            for _ in range(50):
                z2 = float(rng.uniform(z_min, z_max))
                if abs(z2 - z1) > 0.05:
                    break
            N2 = float(sample_logNHI(rng, n=1, beta=NHI_BETA)[0])
            z_dla2[i]  = z2
            logNHI2[i] = N2
        else:
            n_dla[i] = 1

    snr_gu_arr = np.zeros(N, dtype=np.float32)   # 在批量循环中逐谱由 flux 公式计算
    snr_gv_arr = np.zeros(N, dtype=np.float32)
    snr_gi_arr = np.zeros(N, dtype=np.float32)

    n_total_dla = int(has_dla.sum())
    n_2dla      = int((n_dla == 2).sum())
    print(f"DLA statistics (pre-sampled):")
    print(f"  No DLA  : {N - n_total_dla:>6d}  ({(N-n_total_dla)/N*100:.1f}%)")
    print(f"  1 DLA   : {(n_dla==1).sum():>6d}  ({(n_dla==1).sum()/N*100:.1f}%)")
    print(f"  2 DLAs  : {n_2dla:>6d}  ({n_2dla/N*100:.1f}%)\n")

    # 批量生成
    try:
        from tqdm import tqdm
        _tqdm = True
    except ImportError:
        _tqdm = False

    n_batches = (N + BS - 1) // BS
    t_start   = time.time()
    print(f"Starting batch generation ({n_batches} batches)...\n")

    batch_iter = tqdm(range(n_batches), desc="Batches", unit="batch") \
                 if _tqdm else range(n_batches)

    for b in batch_iter:
        i0 = b * BS
        i1 = min(i0 + BS, N)
        nb = i1 - i0
        t_b = time.time()

        # simqso 原始 QSO 光谱
        raw = generate_qso_batch(
            wave, z_qso_arr[i0:i1],
            batch_idx=b, master_seed=args.seed,
        )   # [nb, N_PIX], float64

        if args.n_workers <= 1:
            # Exact original serial processing path. Use this when byte-for-byte
            # reproducibility with the pre-parallel script is more important than speed.
            for j in range(nb):
                gi   = i0 + j
                spec = raw[j]   # float64

                # ── 由 GU band 平均 flux 计算三波段 SNR（公式 9）────────────────
                # log10[SNR_14A] = -0.261 + 0.469 * log10(f / f0)，f0 = 1e-17
                # 然后换算到 8A bin：SNR_8A = SNR_14A * sqrt(8/14)
                # GV ≈ 2×GU，GI ≈ 1.5×GV，各有 SNR_SCATTER 相对散射
                gu_mask   = wave < LCUT_GV
                f_gu_mean = float(np.mean(spec[gu_mask]))
                if f_gu_mean > 0:
                    log_snr14 = -0.261 + 0.469 * np.log10(f_gu_mean / F0) +0.4
                    snr14     = 10.0 ** log_snr14
                    snr_gu    = snr14 * np.sqrt(8.0 / 14.0)
                else:
                    snr_gu = 0.8   # fallback for very faint/failed spectra
                snr_gu = max(snr_gu, 0.05)   # 防止极端小值

                ratio_gv  = SNR_GV_RATIO * (1.0 + rng.normal(0.0, SNR_SCATTER))
                ratio_gi  = SNR_GI_RATIO * (1.0 + rng.normal(0.0, SNR_SCATTER))
                ratio_gv  = max(ratio_gv, 1.5)
                ratio_gi  = max(ratio_gi, 1.5)
                snr_gv    = snr_gu * ratio_gv
                snr_gi    = snr_gv * ratio_gi

                snr_gu_arr[gi] = snr_gu
                snr_gv_arr[gi] = snr_gv
                snr_gi_arr[gi] = snr_gi

                # 插入 DLA（使用原始 dla_spec 接口）
                if has_dla[gi]:
                    dla_list = [(float(z_dla1[gi]), float(logNHI1[gi]))]
                    if n_dla[gi] == 2:
                        dla_list.append((float(z_dla2[gi]), float(logNHI2[gi])))
                    trans = insert_dlas(wave, dla_list)
                else:
                    trans = np.ones(N_PIX, dtype=np.float64)

                spec_dla = spec * trans   # DLA 吸收后

                # FLUX_CLEAN: 只降分辨率（无噪声）
                if flux_clean is not None:
                    spec_dla_clean = np.nan_to_num(
                        spec_dla.astype(np.float64),
                        nan=0.0, posinf=0.0, neginf=0.0)
                    flux_clean[gi] = gaussian_filter1d(
                        spec_dla_clean, sigma=_SIGMA_PIX
                    ).astype(np.float32)

                # FLUX: 降分辨率 + CSST 噪声（严格按原始 pipeline）
                flux_noisy[gi] = filter_like_pipeline(
                    wave, spec_dla,
                    lcut_gv=LCUT_GV, lcut_gi=LCUT_GI,
                    min_sigma=MIN_SIGMA,
                    snr_gu=snr_gu, snr_gv=snr_gv, snr_gi=snr_gi,
                    rng=rng,
                )
        else:
            # Pre-draw all random variates in the same order as the original serial loop.
            # This keeps the generated sample deterministic for a fixed seed even when the
            # per-spectrum processing below is threaded.
            band_masks = [
                wave < LCUT_GV,
                (wave >= LCUT_GV) & (wave < LCUT_GI),
                wave >= LCUT_GI,
            ]
            payloads = []
            for j in range(nb):
                gi = i0 + j
                ratio_gv_draw = rng.normal(0.0, SNR_SCATTER)
                ratio_gi_draw = rng.normal(0.0, SNR_SCATTER)
                noise_norms = [rng.normal(0.0, 1.0, size=mask.sum()) for mask in band_masks]
                payloads.append((
                    gi, raw[j], wave,
                    int(has_dla[gi]), int(n_dla[gi]),
                    z_dla1[gi], logNHI1[gi], z_dla2[gi], logNHI2[gi],
                    ratio_gv_draw, ratio_gi_draw, noise_norms,
                    args.no_clean, LCUT_GV, LCUT_GI, MIN_SIGMA,
                    F0, SNR_GV_RATIO, SNR_GI_RATIO, N_PIX,
                ))

            with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
                results = list(pool.map(process_one_spectrum, payloads))

            for gi, noisy, clean, snr_gu, snr_gv, snr_gi in results:
                flux_noisy[gi] = noisy
                if flux_clean is not None:
                    flux_clean[gi] = clean
                snr_gu_arr[gi] = snr_gu
                snr_gv_arr[gi] = snr_gv
                snr_gi_arr[gi] = snr_gi

        if not _tqdm:
            elapsed = time.time() - t_b
            print(f"  Batch {b+1:>4d}/{n_batches}  [{i0}-{i1-1}]  "
                  f"{elapsed:.1f}s  ({nb/elapsed:.0f} spec/s)")

    total = time.time() - t_start
    print(f"\nGeneration complete: {N} spectra in {total/60:.1f} min "
          f"({N/total:.1f} spec/s)")

    # ══════════════════════════════════════════════════════════════════════════
    # 6.  保存 FITS.gz
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\nWriting {args.output} ...")

    pri = fits.PrimaryHDU()
    h = pri.header
    h["ORIGIN"]   = ("CSST DLA Generator",  "Script that created this file")
    h["N_SPEC"]   = (N,            "Total number of QSO spectra")
    h["ZQSO_MIN"] = (Z_QSO_MIN,   "Min QSO redshift")
    h["ZQSO_MAX"] = (Z_QSO_MAX,   "Max QSO redshift")
    h["DLA_FRAC"] = (DLA_FRAC,    "Fraction of QSOs with DLA")
    h["TWO_FRAC"] = (TWO_FRAC,    "Frac of DLA-QSOs with 2 DLAs")
    h["NHI_MIN"]  = (19.5,        "Min log10(N_HI)")
    h["NHI_MAX"]  = (22.5,        "Max log10(N_HI)")
    h["NHI_BETA"] = (NHI_BETA,    "CDDF power-law slope beta")
    h["LCUT_GV"]  = (LCUT_GV,      "GU/GV boundary [A]")
    h["LCUT_GI"]  = (LCUT_GI,      "GV/GI boundary [A]")
    h["GVRATIO"]  = (SNR_GV_RATIO, "SNR_GV/SNR_GU ratio (mean)")
    h["GIRATIO"]  = (SNR_GI_RATIO, "SNR_GI/SNR_GV ratio (mean)")
    h["SNRSCAT"]  = (SNR_SCATTER,  "Relative scatter on SNR ratios")
    h["MINSIG"]   = (MIN_SIGMA,   "Noise floor min_sigma")
    h["RGRID"]    = (_R_GRID,     "R_GRID used for sigma_pix")
    h["LAMEFF"]   = (_LAM_EFF,    "lam_eff used for sigma_pix [A]")
    h["SIGPIX"]   = (_SIGMA_PIX,  "Gaussian sigma in pixels (fixed)")
    h["SEED"]     = (args.seed,   "Master RNG seed")
    h["NWAVE"]    = (N_PIX,       "Number of wavelength pixels")
    h["WMIN"]     = (float(wave.min()), "Min wavelength [A]")
    h["WMAX"]     = (float(wave.max()), "Max wavelength [A]")

    hdu_wave = fits.ImageHDU(data=wave.astype(np.float32), name="WAVELENGTH")
    hdu_wave.header["BUNIT"] = "Angstrom"
    hdu_wave.header["COMMENT"] = "fixed_R_dispersion(2000, 8000, 2000)"

    hdu_flux = fits.ImageHDU(data=flux_noisy, name="FLUX")
    hdu_flux.header["BUNIT"] = "arbitrary"
    hdu_flux.header["COMMENT"] = "Noisy flux: DLA + resolution degradation + CSST noise"

    hdus = [pri, hdu_wave, hdu_flux]

    if flux_clean is not None:
        hdu_clean = fits.ImageHDU(data=flux_clean, name="FLUX_CLEAN")
        hdu_clean.header["BUNIT"] = "arbitrary"
        hdu_clean.header["COMMENT"] = "Noiseless flux: DLA + resolution degradation only"
        hdus.append(hdu_clean)

    # LABELS: 浮点列缺失值用 NaN（FITS 标准），整型列无 null 问题
    cols = [
        fits.Column(name="Z_QSO",   format="E", array=z_qso_arr),
        fits.Column(name="HAS_DLA", format="I", array=has_dla),
        fits.Column(name="N_DLA",   format="I", array=n_dla),
        fits.Column(name="Z_DLA1",  format="E", array=z_dla1),
        fits.Column(name="LOGNHI1", format="E", array=logNHI1,
                    unit="log10(cm^-2)"),
        fits.Column(name="Z_DLA2",  format="E", array=z_dla2),
        fits.Column(name="LOGNHI2", format="E", array=logNHI2,
                    unit="log10(cm^-2)"),
        fits.Column(name="SNR_GU",  format="E", array=snr_gu_arr),
        fits.Column(name="SNR_GV",  format="E", array=snr_gv_arr),
        fits.Column(name="SNR_GI",  format="E", array=snr_gi_arr),
    ]
    hdu_lab = fits.BinTableHDU.from_columns(cols, name="LABELS")
    hdu_lab.header["COMMENT"] = "Training labels. Absent DLA => Z_DLA/LOGNHI = NaN."
    hdus.append(hdu_lab)

    hdul = fits.HDUList(hdus)
    hdul.writeto(args.output, overwrite=True)

    fsize_mb = os.path.getsize(args.output) / 1e6
    print(f"Done!  {args.output}  ({fsize_mb:.1f} MB)\n")
    print("Extensions:")
    for hdu in hdul:
        shape = getattr(hdu.data, 'shape', 'N/A') if hdu.data is not None else 'N/A'
        print(f"  [{hdu.name:12s}]  {str(shape)}")


# ══════════════════════════════════════════════════════════════════════════════
# 本地调试：不需要 simqso 的 mock（仅测流程用）
# ══════════════════════════════════════════════════════════════════════════════

def _mock_qso_batch(wave, n, seed=42):
    rng = np.random.default_rng(seed)
    spectra = []
    for _ in range(n):
        cont = (wave / 4000.0) ** (-1.5)
        forest = np.ones(len(wave))
        m = wave < 4000.0
        forest[m] = np.exp(-rng.exponential(0.1, size=m.sum()))
        spec = cont * forest * (1.0 + 0.05 * rng.normal(size=len(wave)))
        spectra.append(spec.clip(0))
    return np.array(spectra)


if __name__ == "__main__":
    main()
