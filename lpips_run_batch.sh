#!/usr/bin/env bash

# v10 v13   # 0.85
# v11 v15   # 0.90
# v12 v16   # 0.95
# v17 v22   # 0.80
# v18 v19   # 0.70
# v20 v21   # 0.75
# v23 v24   # 0.65
# GROUPS


set -euo pipefail

KEEP_RATIOS=(0.5)
DEL_SEEDS=(66 77 88)
MATCH_SEEDS=(44)
DATASETS=(odsr)
METRICS=(dreamsim)
SAMPLING="random"

# 한 줄 = 한 그룹 ("v10 v13" 형식, 뒤에 주석 허용)
while IFS= read -r LINE; do
  # 빈줄 스킵
  [[ -z "${LINE//[[:space:]]/}" ]] && continue
  # 주석 제거
  GROUP="${LINE%%#*}"
  # 양끝 공백 제거
  GROUP="$(echo "$GROUP" | xargs)" || true
  [[ -z "$GROUP" ]] && continue

  # 단어 수 = match_ratio
  read -r -a VER_ARR <<< "$GROUP"
  MR="${#VER_ARR[@]}"

  echo "▶ versions=($GROUP)  mode=vs  k=$MR"

  for KR in "${KEEP_RATIOS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      for METRIC in "${METRICS[@]}"; do
        for DSEED in "${DEL_SEEDS[@]}"; do
          for MSEED in "${MATCH_SEEDS[@]}"; do
            python run_pipeline.py \
              --dataset     "$DATASET" \
              --version     "$GROUP" \
              --ratio       "$KR" \
              --sampling    "$SAMPLING" \
              --match-mode  vs \
              --match-ratio "$MR" \
              --seed-del    "$DSEED" \
              --seed-match  "$MSEED" \
              --metric      "$METRIC"
              # --quality  --save-results 등 필요시 추가
          done
        done
      done
    done
  done
done <<'GROUPS'
v10 v13   # 0.85
v11 v15   # 0.90
v12 v16   # 0.95
v17 v22   # 0.80
v18 v19   # 0.70
v20 v21   # 0.75
v23 v24   # 0.65
GROUPS

