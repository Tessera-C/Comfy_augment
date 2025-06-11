#!/usr/bin/env bash
set -e

VERSIONS="v9,v10"
RATIO=0.6
SEED=42

# 명령을 배열로 보관
BASE=(python run_pipeline.py
      --version "$VERSIONS"
      --ratio   "$RATIO"
      --seed    "$SEED"
)   # --skip-train 를 넣고 싶으면 이 배열 끝에 추가

MATCH_RATIOS=(0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0)

for MR in "${MATCH_RATIOS[@]}"; do
    echo "▶ match-ratio=$MR"
    "${BASE[@]}" --match-ratio "$MR"
done