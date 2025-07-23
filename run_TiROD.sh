#!/usr/bin/env bash
set -e

# --- 기존 설정 ---
# VERSIONS="v9, v10, v11, v12, v13"
# RATIO=0.70
# SEED=49
#
# # 명령을 배열로 보관
# BASE=(python run_pipeline.py
#       --version "$VERSIONS"
#       --ratio   "$RATIO"
#       --seed    "$SEED"
# )   # --skip-train 를 넣고 싶으면 이 배열 끝에 추가
#
# MATCH_RATIOS=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0)
# # MATCH_RATIOS=(2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0)
#
# for MR in "${MATCH_RATIOS[@]}"; do
#     echo "▶ match-ratio=$MR"
#     "${BASE[@]}" --match-ratio "$MR"
# done

# --- 새로운 설정 ---
VERSIONS="v10, v11"
RATIO=0
SEEDS=(50)

# MATCH_RATIOS=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0)
# MATCH_RATIOS=(4.0 4.25 4.5 4.75 5.0)
MATCH_RATIOS=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2)


for SEED in "${SEEDS[@]}"; do
    echo "▶ Running for SEED: $SEED"
    BASE=(python run_pipeline_tirod.py
          --version "$VERSIONS"
          --ratio   "$RATIO"
          --seed    "$SEED"
    )   # --skip-train 를 넣고 싶으면 이 배열 끝에 추가

    for MR in "${MATCH_RATIOS[@]}"; do
        echo "▶ match-ratio=$MR"
        "${BASE[@]}" --match-ratio "$MR"
    done
done