#!/usr/bin/env bash
# Fill missing raw_trials/ for p=0.2/0.4/0.6/0.8/1.0 on the 41×41 grid
# (K step 0.05, δ step 0.025). Existing npz files are skipped.
set -euo pipefail
cd "$(dirname "$0")"

python3 code/generate_raw_grid.py --trials 1 "$@"
