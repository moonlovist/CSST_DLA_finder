#!/bin/bash
set -euo pipefail

TEST_FITS="${TEST_FITS:-./data/test.fits.gz}"
CHECKPOINT="${CHECKPOINT:-./runs/run01/best_model.pt}"
OUTPUT_FITS="${OUTPUT_FITS:-./submissions/team_submission.fits}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"
MIN_LOG_NHI="${MIN_LOG_NHI:-20.3}"
STRIDE="${STRIDE:-8}"
WINDOW_SIZE="${WINDOW_SIZE:-256}"
TOP_K="${TOP_K:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MERGE_SEPARATION_PIX="${MERGE_SEPARATION_PIX:-80}"

mkdir -p "$(dirname "${OUTPUT_FITS}")"

python build_submission_window_cnn.py \
  --test_fits "${TEST_FITS}" \
  --checkpoint "${CHECKPOINT}" \
  --output_fits "${OUTPUT_FITS}" \
  --stride "${STRIDE}" \
  --window_size "${WINDOW_SIZE}" \
  --top_k "${TOP_K}" \
  --batch_size "${BATCH_SIZE}" \
  --merge_separation_pix "${MERGE_SEPARATION_PIX}" \
  --confidence_threshold "${CONFIDENCE_THRESHOLD}" \
  --min_log_nhi "${MIN_LOG_NHI}"
