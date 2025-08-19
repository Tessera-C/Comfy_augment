#!/usr/bin/env bash
# -------------------------------------------------------------------
#  다양한 조합으로 run_pipeline.py 반복 실행
#
#  • VERSIONS      : 공백 구분 → 한 번에 전달(v9 v10 … → "v9 v10 …")
#  • MATCH_RATIOS  : 한 실험 당 match-ratio 값(실수 또는 정수)들
#  • MATCH_MODE    : mix(기존) | vs(verselect)
#  • DEL_SEEDS     : 원본 삭제 시드(여러 개 가능)
#  • MATCH_SEEDS   : 생성 이미지 매칭 시드(여러 개 가능)
#  • DATASETS      : odsr / tirod 조합
#  • METRICS       : lpips / dreamsim …
# -------------------------------------------------------------------
set -e

# ── 사용자 설정 ─────────────────────────────────────────────────────
# VERSIONS=(v10 v10 v11 v12 v13)      # 한 실험에 투입할 버전 목록
VERSIONS=(v100 v101 v102 v103 v104 v105 v106 v107 v108 v109 v110 v111)
MATCH_RATIOS=(0 1 2 3 4 5 6 7 8)
MATCH_RATIOS=(0)               # mix → 실수 허용, vs → 정수(k개 버전)
MATCH_MODE="vs"                    # mix | vs
DEL_SEEDS=(24 25 26)                     # 여러 개면 반복
MATCH_SEEDS=(65)                # 여러 개면 반복
DATASETS=(odsr)              # 복수 선택 가능
METRICS=(dreamsim)                 # dreamsim, lpips 등
SAMPLING="random"                  # tirod 전용 interval 원하면 "interval"
KEEP_RATIO=0.7                     # --ratio (원본 남길 비율)

# ── 내부 편의 함수 ─────────────────────────────────────────────────
join_by() { local IFS="$1"; shift; echo "$*"; }

VER_ARG=$(join_by " " "${VERSIONS[@]}")   # 공백 구분 (run_pipeline 허용)

# ── 실행 루프 ───────────────────────────────────────────────────────
for DATASET in "${DATASETS[@]}";   do
  for METRIC  in "${METRICS[@]}";  do
    for DSEED  in "${DEL_SEEDS[@]}";  do
      for MSEED  in "${MATCH_SEEDS[@]}"; do
        for MR in "${MATCH_RATIOS[@]}";  do

          # vs 모드일 때 MR > 버전 수면 건너뜀
          if [[ "$MATCH_MODE" == "vs" && "$MR" -gt "${#VERSIONS[@]}" ]]; then
            echo "[SKIP] vs 모드: match_ratio($MR) > 버전 수(${#VERSIONS[@]})"
            continue
          fi

          echo "▶ dataset=$DATASET  versions=(${VER_ARG// /, })  mode=$MATCH_MODE  m_ratio=$MR  del_seed=$DSEED  match_seed=$MSEED  metric=$METRIC"

          python run_pipeline.py \
            --dataset      "$DATASET"      \
            --version      "$VER_ARG"      \
            --ratio        "$KEEP_RATIO"   \
            --sampling     "$SAMPLING"     \
            --match-mode   "$MATCH_MODE"   \
            --match-ratio  "$MR"           \
            --seed-del     "$DSEED"        \
            --seed-match   "$MSEED"        \
            --metric       "$METRIC"

        done
      done
    done
  done
done
