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

VERSIONS="v9,v10,v11,v12,v13"
RATIO=0.75
SEED=66

# 공통 옵션
COMMON_OPTS=(
  --version "$VERSIONS"
  --ratio   "$RATIO"
  --seed    "$SEED"
  --quality               # LPIPS 계산 + 통계
  --save-results          # 학습 완료 후 결과 복사
)

MATCH_RATIOS=(1.0 2.0 3.0 4.0 5.0)

for MR in "${MATCH_RATIOS[@]}"; do
  echo "──────────────────────────────────────────"
  echo "▶ match-ratio=$MR  |  LPIPS=TOP 50 %"
  python run_pipeline.py \
    "${COMMON_OPTS[@]}" \
    --match-ratio "$MR" \
    --lpips-mode top --lpips-percent 50

  echo "------------------------------------------"
  echo "▶ match-ratio=$MR  |  LPIPS=BOTTOM 50 %"
  python run_pipeline.py \
    "${COMMON_OPTS[@]}" \
    --match-ratio "$MR" \
    --lpips-mode bottom --lpips-percent 50
done
