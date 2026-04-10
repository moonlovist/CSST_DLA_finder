#!/bin/bash
set -euo pipefail

TRAIN_FITS="${TRAIN_FITS:-./data/train.fits.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/run01}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
JITTER_PIX="${JITTER_PIX:-16}"
MIN_POSITIVE_LOG_NHI="${MIN_POSITIVE_LOG_NHI:-20.3}"
NUM_NEG_PER_SPEC="${NUM_NEG_PER_SPEC:-8}"
NUM_HARD_NEG_PER_DLA="${NUM_HARD_NEG_PER_DLA:-6}"

python train_window_cnn.py \
  --train_fits "${TRAIN_FITS}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --jitter_pix "${JITTER_PIX}" \
  --min_positive_log_nhi "${MIN_POSITIVE_LOG_NHI}" \
  --num_neg_per_spec "${NUM_NEG_PER_SPEC}" \
  --num_hard_neg_per_dla "${NUM_HARD_NEG_PER_DLA}" \
  --amp
