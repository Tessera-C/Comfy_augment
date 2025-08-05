#!/usr/bin/env bash
# ---------------------------------------------------------------
#  match-ratio = 1 로 고정해 여러 버전을 자동 실험
#
#  • VERSIONS      : 공백 구분 배열로 나열 (v9 v10 …)
#  • MATCH_SEEDS   : 생성 이미지 매칭 시드 (여러 개면 반복 실행)
#  • DATASETS      : odsr / tirod 중 선택·조합
#  • METRICS       : dreamsim / lpips 등 품질 지표
# ---------------------------------------------------------------
set -e

# ── 사용자 설정 ───────────────────────────────────────────────────────
# VERSIONS=(v10 v11 v12 v13 v14 v15 v16 v17)
VERSIONS=(v17)          # 실행할 버전들
MATCH_RATIO=1                      # 고정 (수정하려면 여기만 변경)

DEL_SEED=34                        # 원본 삭제 시드
MATCH_SEEDS=(34)       # 매칭 시드 (반복 실행용)

DATASETS=(odsr)              # 대상 데이터셋
METRICS=(dreamsim)                 # dreamsim 만 쓰면 (dreamsim)
SAMPLING="random"                  # tirod 에서 interval 원하면 "interval"
KEEP_RATIO=0.5                     # --ratio (원본 70 %, 삭제 30 %)

# ── 실행 루프 ─────────────────────────────────────────────────────────
for VER in "${VERSIONS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for METRIC in "${METRICS[@]}"; do
      for MS in "${MATCH_SEEDS[@]}"; do

        echo "▶ ver=${VER} dataset=${DATASET} metric=${METRIC} mseed=${MS}"

        python run_pipeline.py \
            --dataset      "$DATASET"      \
            --version      "$VER"          \
            --ratio        "$KEEP_RATIO"   \
            --sampling     "$SAMPLING"     \
            --match-ratio  "$MATCH_RATIO"  \
            --seed-del     "$DEL_SEED"     \
            --seed-match   "$MS"           \
            --quality                       \
            --metric       "$METRIC"

      done
    done
  done
done
