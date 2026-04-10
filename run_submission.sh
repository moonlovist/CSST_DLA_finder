#!/bin/bash
set -euo pipefail

TEST_FITS="${TEST_FITS:-./data/test.fits.gz}"
CHECKPOINT="${CHECKPOINT:-./runs/run01/best_model.pt}"
OUTPUT_FITS="${OUTPUT_FITS:-./submissions/team_submission.fits}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.5}"
MIN_LOG_NHI="${MIN_LOG_NHI:-20.3}"

mkdir -p "$(dirname "${OUTPUT_FITS}")"

python build_submission_fits.py \
  --test_fits "${TEST_FITS}" \
  --checkpoint "${CHECKPOINT}" \
  --output_fits "${OUTPUT_FITS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --confidence_threshold "${CONFIDENCE_THRESHOLD}" \
  --min_log_nhi "${MIN_LOG_NHI}"
