#!/usr/bin/env bash
# Resolution sweep orchestrator (overnight, unattended).
#
# Builds the 256/512/1024-bin comparison against the LDA (1024-bin PSD) baseline
# across three studies: baseline accuracy, interference transfer, window-length.
# 256-bin results already exist; 512-bin baseline is trained separately. This
# script fills in the rest.
#
# RUN AFTER the 512-bin baseline training finishes, to avoid CPU contention:
#   bash run_resolution_sweep.sh
#
# Rough wall-clock on this CPU box: ~4.5 h (1024 extract ~40m, 1024 train ~3h,
# transfers ~45m, sweeps a few min).

set -euo pipefail
cd "$(dirname "$0")"   # project root

stamp() { date '+%H:%M:%S'; }

# Wait for the separately-running 512-bin baseline to finish (its last artifact
# is cnn_embeddings_512.npy) before starting, to avoid CPU contention. This lets
# the whole sweep be launched now (one approval) yet start only once the CPU is free.
echo "[$(stamp)] waiting for 512-bin baseline training to finish..."
while [ ! -f CNN/results/cnn_embeddings_512.npy ]; do
  sleep 60
done
echo "[$(stamp)] 512-bin baseline done — CPU free, starting sweep"

echo "[$(stamp)] === [1/6] extract 1024-bin spectrograms ==="
python CNN/scripts/extract_spectrograms.py 1024

echo "[$(stamp)] === [2/6] train 1024-bin baseline CNN ==="
python CNN/scripts/train_cnn.py 1024

echo "[$(stamp)] === [3/6] interference transfer @512 ==="
python verify/scripts/cnn_interference_transfer.py 512

echo "[$(stamp)] === [4/6] interference transfer @1024 ==="
python verify/scripts/cnn_interference_transfer.py 1024

echo "[$(stamp)] === [5/6] window-length sweep @512 ==="
python verify/scripts/segment_length_sweep.py 512

echo "[$(stamp)] === [6/6] window-length sweep @1024 ==="
python verify/scripts/segment_length_sweep.py 1024

echo "[$(stamp)] === resolution sweep complete ==="
