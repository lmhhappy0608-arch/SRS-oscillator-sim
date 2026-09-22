#!/usr/bin/env bash
# 小规模试跑：4 个 (K, delta) 格点
# --benchmark：N=100, T=1000（秒级完成，验证流程）
# 去掉 --benchmark 则为正式参数 N=1000, T=10000（每格约十余分钟）
# 若 build.sh 已启用 OpenMP，可加 --workers N 或 OMP_NUM_THREADS=N 并行跑格点
set -euo pipefail
cd "$(dirname "$0")"
./build.sh
./build/generate_raw_grid \
  --benchmark \
  --k-min 0 --k-max 0.4 --k-step 0.4 \
  --delta-min 0.10 --delta-max 0.15 --delta-step 0.05 \
  --p 0.8 --trials 1 \
  --output-dir raw_trials_cpp
echo "完成。输出: raw_trials_cpp/"
