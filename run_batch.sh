#!/usr/bin/env bash
set -e

VERSIONS="v9, v10, v11, v12, v13"
RATIO=0.75
SEED=66

# 명령을 배열로 보관
BASE=(python run_pipeline.py
      --version "$VERSIONS"
      --ratio   "$RATIO"
      --seed    "$SEED"
)   # --skip-train 를 넣고 싶으면 이 배열 끝에 추가

MATCH_RATIOS=(3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0)
# MATCH_RATIOS=(2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0)

for MR in "${MATCH_RATIOS[@]}"; do
    echo "▶ match-ratio=$MR"
    "${BASE[@]}" --match-ratio "$MR"
done