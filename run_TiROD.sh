# #!/usr/bin/env bash
# set -e

# # --- 새로운 설정 ---
# VERSIONS="v10, v11, v12, v13, v14, v15"
# RATIO=0.7
# SEEDS=(11 22 33)

# MATCH_RATIOS=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 5.25 5.5 5.75 6)
# # MATCH_RATIOS=(4.0 4.25 4.5 4.75 5.0)
# # MATCH_RATIOS=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2)


# for SEED in "${SEEDS[@]}"; do
#     echo "▶ Running for SEED: $SEED"
#     BASE=(python run_pipeline_tirod.py
#           --version "$VERSIONS"
#           --ratio   "$RATIO"
#           --seed    "$SEED"
#     )   # --skip-train 를 넣고 싶으면 이 배열 끝에 추가

#     for MR in "${MATCH_RATIOS[@]}"; do
#         echo "▶ match-ratio=$MR"
#         "${BASE[@]}" --match-ratio "$MR"
#     done
# done

#!/usr/bin/env bash
set -e

# --- 새로운 설정 ---
VERSIONS="v10, v11, v12, v13, v14, v15"
RATIO=0.1
SEEDS=(40)
DATASETS=(tirod)   # 실행 대상 데이터셋 (odsr tirod)
# METRICS=(dreamsim)      # 배열로 두면 LPIPS 함께 돌릴 때 METRICS=(dreamsim lpips)
SAMPLING="interval"       # tirod 에서 interval 원하면 여기만 "interval" 로

MATCH_RATIOS=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0 5.25 5.5 5.75 6)
# MATCH_RATIOS=(4.0 4.25 4.5 4.75 5.0)
# MATCH_RATIOS=(0 0.25 0.5 0.75 1 1.25 1.5 1.75 2)


for SEED in "${SEEDS[@]}"; do
    echo "▶ Running for SEED: $SEED"
    BASE=(python run_pipeline.py
          --version "$VERSIONS"
          --ratio   "$RATIO"
          --seed    "$SEED"
          --dataset "tirod"
          --sampling "$SAMPLING"
    )   # --skip-train 를 넣고 싶으면 이 배열 끝에 추가

    for MR in "${MATCH_RATIOS[@]}"; do
        echo "▶ match-ratio=$MR"
        "${BASE[@]}" --match-ratio "$MR"
    done
done