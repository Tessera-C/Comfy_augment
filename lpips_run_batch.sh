#!/usr/bin/env bash
# ------------------------------------------------------------------
#  ▪ 버전   : v9 ~ v13   (5개)
#  ▪ 삭제율 : 0.75       (원본 25 %)
#  ▪ match-ratio : 1.0, 2.0, 3.0, 4.0, 5.0   (5단계)
#  ▪ LPIPS  : TOP 50 % · BOTTOM 50 % (각 단계마다 2회)
#  ─▶ 총 10회 학습
#
#  실행 전 : chmod +x lpips_run_batch.sh
#            ./lpips_run_batch.sh
# ------------------------------------------------------------------
set -e

DEL_SEED=10        # 원본 삭제용 고정 시드
MATCH_RATIO=2.5     # 접두어 매칭 비율
MATCH_SEEDS=(6 7 8 9 10)   # 접두어 매칭 시드를 바꿀 값들

for SM in "${MATCH_SEEDS[@]}"; do
  echo "▶ seed_match=$SM  TOP50"
  python run_pipeline.py \
      --version v9,v10,v11,v12,v13 \
      --ratio 0.75                \
      --match-ratio "$MATCH_RATIO" \
      --seed-del  "$DEL_SEED"     \
      --seed-match "$SM"          \
      --quality                   \
      --lpips-mode top --lpips-percent 50

  echo "▶ seed_match=$SM  BOTTOM50"
  python run_pipeline.py \
      --version v9,v10,v11,v12,v13 \
      --ratio 0.75                \
      --match-ratio "$MATCH_RATIO" \
      --seed-del  "$DEL_SEED"     \
      --seed-match "$SM"          \
      --quality                   \
      --lpips-mode bottom --lpips-percent 50
done
