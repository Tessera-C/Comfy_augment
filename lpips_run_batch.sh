#!/usr/bin/env bash
# ------------------------------------------------------------------
#  ▪ 버전        : v9 ~ v13   (5개)
#  ▪ 삭제율      : 0.75       (원본 25 % 유지)
#  ▪ match-ratio : 1.0, 2.0, 3.0, 4.0, 5.0   (5단계)
#  ▪ DreamSim    : TOP 50 % · BOTTOM 50 %    (각 단계마다 2회)
#  ─▶ 총 10회 학습
#
#  실행 전 :  chmod +x dreamsim_run_batch.sh
#             ./dreamsim_run_batch.sh
# ------------------------------------------------------------------
set -e

DEL_SEED=10                  # 원본 삭제용 고정 시드
MATCH_RATIOS=(1 2 3 4 5)     # 접두어 매칭 비율들
MATCH_SEEDS=(6 7 8 9 10)     # 접두어 매칭 시드
METRIC="dreamsim"            # or lpips

for R in "${MATCH_RATIOS[@]}"; do
  for SM in "${MATCH_SEEDS[@]}"; do
    echo "▶ match_ratio=$R  seed_match=$SM  TOP50"
    python run_pipeline.py \
        --version v9,v10,v11,v12,v13 \
        --ratio 0.75 \
        --match-ratio "$R" \
        --seed-del  "$DEL_SEED" \
        --seed-match "$SM" \
        --quality \
        --metric "$METRIC" \
        --lpips-mode top --lpips-percent 50

    echo "▶ match_ratio=$R  seed_match=$SM  BOTTOM50"
    python run_pipeline.py \
        --version v9,v10,v11,v12,v13 \
        --ratio 0.75 \
        --match-ratio "$R" \
        --seed-del  "$DEL_SEED" \
        --seed-match "$SM" \
        --quality \
        --metric "$METRIC" \
        --lpips-mode bottom --lpips-percent 50
  done
done