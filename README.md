# CSST_DLA_finder

Baseline code package for the CSST DLA Finder Challenge.

This folder is structured so it can be pushed to GitHub and cloned on NERSC, then run directly on a GPU interactive node.

## What Is Included

- `generate_train_fits.py`
  Generate the labeled training FITS file.
- `generate_test_fits.py`
  Generate the blind test FITS file with `META(TARGETID, Z_QSO)`.
- `dla_cnn.py`
  Shared data loading, preprocessing, and the 1D CNN model.
- `train_dla_cnn.py`
  Train the CNN baseline.
- `predict_dla_cnn.py`
  Write a debugging CSV for blind-test predictions.
- `build_submission_fits.py`
  Build the official submission FITS by appending a `RESULTS` extension.
- `validate_csst_dla_files.py`
  Validate train, test, and submission FITS files against the challenge schema.
- `run_train.sh`
  Simple training entrypoint for local or NERSC use.
- `run_submission.sh`
  Simple submission-building entrypoint.
- `requirements.txt`
  Minimal Python dependencies.
- `MODEL.md`
  Short model description for the competition code package.

## Corrections Applied After Reading The Official Docs

The official documents in the parent directory imply several required corrections relative to the earlier scripts:

1. The blind test file should contain `META`, not `LABELS`.
2. `META` should include `TARGETID` and `Z_QSO`.
3. The final deliverable should be a FITS file with a `RESULTS` extension, not only a CSV.

These corrections are implemented in this folder.

## Challenge Compliance Status

This package now covers the required deliverables listed in the challenge note:

- training code
- model definition
- inference code
- official FITS submission builder
- short model description

The submission builder writes the required fields:

- `TARGETID`
- `Z_QSO`
- `Z_DLA`
- `LOG_NHI`
- `CONFIDENCE`

The package also includes a schema validator to catch formatting mistakes before submission.

## Important Note About The Labels

The generator samples `log N_HI` in `[19.5, 22.5]`, while the challenge document describes DLA scientifically as `log N_HI >= 20.3`.

This may be intentional if the organizers want sub-DLAs included in training, but it is still worth confirming. To keep the final catalog conservative, `build_submission_fits.py` defaults to:

- `--min_log_nhi 20.3`

## Recommended Repository Layout On NERSC

```text
CSST_DLA_finder/
data/
  train.fits.gz
  test.fits.gz
runs/
submissions/
```

## Quick Start On NERSC

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd CSST_DLA_finder
```

### 2. Activate an environment

If you already have a working NERSC environment, use it. Otherwise:

```bash
module load python
python -m pip install --user -r requirements.txt
```

If `torch` is already available in your conda environment, that is preferred.

### 3. Validate the FITS files

```bash
python validate_csst_dla_files.py --train_fits ./data/train.fits.gz
python validate_csst_dla_files.py --test_fits ./data/test.fits.gz
```

### 4. Train

```bash
TRAIN_FITS=./data/train.fits.gz OUTPUT_DIR=./runs/run01 ./run_train.sh
```

### 5. Build the final submission FITS

```bash
TEST_FITS=./data/test.fits.gz \
CHECKPOINT=./runs/run01/best_model.pt \
OUTPUT_FITS=./submissions/team_submission.fits \
./run_submission.sh
```

### 6. Validate the submission

```bash
python validate_csst_dla_files.py --submission_fits ./submissions/team_submission.fits
```

## NERSC Interactive Example

```bash
qinteractive -C gpu -q interactive -t 04:00:00 -A <your_account>
module load python
cd CSST_DLA_finder
python -m pip install --user -r requirements.txt
TRAIN_FITS=./data/train.fits.gz OUTPUT_DIR=./runs/run01 ./run_train.sh
TEST_FITS=./data/test.fits.gz CHECKPOINT=./runs/run01/best_model.pt OUTPUT_FITS=./submissions/team_submission.fits ./run_submission.sh
```

## Current Scope

This is a runnable baseline, not yet an optimized competition solution. The most likely next improvements are:

- sliding-window or anchor-based DLA localization
- uncertainty calibration
- SNR-aware sampling or weighted losses
- model ensembling
