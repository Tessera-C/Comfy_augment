#!/usr/bin/env bash
# ------------------------------------------------------------------
#  ◼ 버전 목록      : v9 ~ v14  (6개, 필요에 맞게 수정)
#  ◼ 삭제율         : 0.75      (원본 25 % 유지)
#  ◼ match-ratio    : 1‥5       (5단계)
#  ◼ 품질 지표      : DreamSim / LPIPS
#  ◼ 데이터셋       : ODSR-IHS, TiROD   ← 둘 다 실행
#  ◼ Split          : TOP 50 % · BOTTOM 50 %
#  → 총   2(지표) × 2(Top/Bottom) × 5(match) × 2(dataset) = 40회 학습
#
#  실행 전 :  chmod +x batch_run.sh
#             ./batch_run.sh
# ------------------------------------------------------------------
set -e

DEL_SEED=20             # 원본 삭제용 고정 시드
MATCH_RATIOS=(6)
MATCH_SEEDS=(7 8 9 10 11 12)

DATASETS=(odsr)   # 실행 대상 데이터셋 (odsr tirod)
METRICS=(dreamsim)      # 배열로 두면 LPIPS 함께 돌릴 때 METRICS=(dreamsim lpips)
SAMPLING="random"       # tirod 에서 interval 원하면 여기만 "interval" 로

VERSIONS="v9,v10,v11,v12,v13,v14,v15"

for DATASET in "${DATASETS[@]}"; do
  for METRIC in "${METRICS[@]}"; do
    for R in "${MATCH_RATIOS[@]}"; do
      for SM in "${MATCH_SEEDS[@]}"; do
        for SPLIT in top bottom; do
          echo "▶ dataset=${DATASET}  metric=${METRIC}  match=${R}  seed=${SM}  ${SPLIT}50"
          python run_pipeline.py \
              --dataset "$DATASET" \
              --version "$VERSIONS" \
              --ratio 0.7 \
              --sampling "$SAMPLING" \
              --match-ratio "$R" \
              --seed-del  "$DEL_SEED" \
              --seed-match "$SM" \
              --quality \
              --metric "$METRIC" \
              --lpips-mode "$SPLIT" --lpips-percent 50
        done
      done
    done
  done
done