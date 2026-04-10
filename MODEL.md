# Model Summary

## Task

This baseline addresses the three core challenge outputs:

- DLA detection
- number of DLAs per spectrum
- regression of `z_DLA` and `log N_HI`

## Inputs

- `FLUX`: noisy CSST-like observed spectrum
- `Z_QSO`
- `SNR_GU`
- `SNR_GV`
- `SNR_GI`

If the blind test file does not provide SNR columns, they are estimated from the flux directly.

## Preprocessing

- Per-spectrum robust normalization using the 95th percentile of `|flux|`
- Auxiliary features standardized with training-set mean and standard deviation
- DLA labels sorted by redshift so the network learns a stable ordering

## Network

- 1D CNN backbone with residual convolutional blocks
- global average pooling over wavelength
- small MLP for auxiliary metadata
- joint head for:
  - binary `HAS_DLA`
  - 3-class `N_DLA` in `{0, 1, 2}`
  - 4 regression outputs: `Z_DLA1`, `LOGNHI1`, `Z_DLA2`, `LOGNHI2`

## Training

- optimizer: `AdamW`
- scheduler: cosine annealing
- loss:
  - binary cross entropy for `HAS_DLA`
  - cross entropy for `N_DLA`
  - masked smooth L1 for DLA parameters

## Output And Submission

- `predict_dla_cnn.py` writes a debugging CSV
- `build_submission_fits.py` writes the official FITS submission with:
  - `TARGETID`
  - `Z_QSO`
  - `Z_DLA`
  - `LOG_NHI`
  - `CONFIDENCE`

## Current Scope

This is a baseline designed for reproducibility and direct execution on NERSC. It is not yet optimized for best challenge score. The most likely future upgrades are:

- sliding-window or anchor-based DLA localization
- explicit uncertainty calibration
- SNR-aware sampling or loss weighting
- ensembling
