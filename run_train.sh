#!/bin/bash
set -euo pipefail

TRAIN_FITS="${TRAIN_FITS:-./data/train.fits.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-./runs/run01}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"

python train_dla_cnn.py \
  --train_fits "${TRAIN_FITS}" \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --num_workers "${NUM_WORKERS}" \
  --amp
