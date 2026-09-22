#!/usr/bin/env bash
# Small smoke test: 4 (K, delta) cells.
# --benchmark: N=100, T=1000 (seconds; pipeline check).
# Drop --benchmark for production N=1000, T=10000 (about ten minutes per cell).
# If build.sh enabled OpenMP, add --workers N or OMP_NUM_THREADS=N.
set -euo pipefail
cd "$(dirname "$0")"
./build.sh
./build/generate_raw_grid \
  --benchmark \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --output-dir raw_trials_cpp
echo "done. output: raw_trials_cpp/"
