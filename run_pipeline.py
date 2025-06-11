#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 데이터셋 준비 & 학습 파이프라인
- 여러 버전 지원, match_ratio≥1.0 시 다중 버전 랜덤 섞기
"""
import argparse
import os
import sys
import datetime
import shutil
from typing import List

# ★ NEW : 공용 유틸 모듈 import
from data_pipeline.utils_misc import (
    TeeWithTimestamp,
    get_auto_log_filename,
    rename_latest_yolo_run,
)

from data_pipeline.preprocess import (
    add_prefix_to_filenames,
    copy_and_prune_dataset,
    copy_augmented_files,
)
from data_pipeline.config_utils import create_yaml, update_data_yaml_in_script
from data_pipeline.runner import run_train_script


# ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 준비 및 훈련 파이프라인")

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="버전 태그. 하나 또는 여러 개를 콤마/스페이스로 구분해 지정 (예: v9,v10)",
    )
    parser.add_argument("--ratio", type=float, default=0.5, help="원본 데이터 삭제 비율 (0.0~1.0)")
    parser.add_argument("--match-ratio", type=float, default=1.0, help="보조 데이터 복사 비율 (0.0~)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--prefix", action="store_true", help="접두어 추가 수행 여부")
    parser.add_argument("--skip-train", action="store_true", help="YOLO 학습 단계 건너뛰기")
    parser.add_argument("--only-train", action="store_true", help="데이터 준비 없이 YOLO 학습만 실행")

    args = parser.parse_args()

    # 여러 버전 파싱 → 리스트
    versions = [v.strip() for v in args.version.replace(",", " ").split() if v.strip()]
    if not versions:
        sys.exit("ERROR: --version 인자를 하나 이상 지정하세요.")

    delete_ratio = args.ratio
    match_ratio = args.match_ratio
    random_seed = args.seed

    version_tag = f"{'-'.join(versions)}_r{int(delete_ratio*100)}_m{int(match_ratio*100)}"

    # 로그 설정
    log_file = get_auto_log_filename(version_tag)
    sys.stdout = TeeWithTimestamp(log_file)
    sys.stderr = sys.stdout
    print(f"[INFO] 로그 저장 위치: {log_file}")

    # 공통 경로
    base_dir = "/home/jhcha/jh_ws/yolo/datasets"
    yolo_script = "/home/jhcha/jh_ws/yolo/yolo_train_ODSR_half.py"
    yaml_out_dir = "/home/jhcha/jh_ws/yolo"
    train_dir = os.path.join(base_dir, f"ODSR-IHS_{version_tag}", "train")

    # ───── only-train 모드 ─────
    if args.only_train:
        print("[!] 데이터 준비를 생략하고 YOLO 학습만 실행 (--only-train)")
        update_data_yaml_in_script(yolo_script, f"{version_tag}.yaml")
        if not args.skip_train:
            rc = run_train_script(yolo_script, yaml_out_dir)
            if rc == 0:
                rename_latest_yolo_run(yaml_out_dir, version_tag)
            sys.exit(rc)
        else:
            print("[!] 학습 실행도 건너뜀 (--skip-train)")
            sys.exit(0)

    # 1. (옵션) 버전별 접두어 생성
    if args.prefix:
        for v in versions:
            add_prefix_to_filenames(v, "output/ODSR_anno", f"output/ODSR_{v}_anno")
            add_prefix_to_filenames(v, "output/ODSR", f"output/ODSR_{v}")

    # 2. 원본 복사 + 일부 삭제 (사본 폴더명에 version_tag 사용)
    print("[2] 데이터 복사 및 삭제...")
    copy_and_prune_dataset(base_dir, versions[0], delete_ratio, seed=random_seed, version_tag=version_tag)

    # 3. 증강본 복사 (다중 버전 섞기 지원)
    print("[3] 접두어가 붙은 파일 복원...")
    copy_augmented_files(train_dir, versions, match_ratio=match_ratio, seed = random_seed)

    # 4. YAML 생성
    print("[4] YAML 생성...")
    create_yaml(version_tag, delete_ratio, match_ratio, save_path=yaml_out_dir)

    # 5. 학습 스크립트 YAML 경로 수정
    print("[5] 학습 스크립트 YAML 경로 수정...")
    update_data_yaml_in_script(yolo_script, f"{version_tag}.yaml")

    # 6. (옵션) 훈련 실행
    if not args.skip_train:
        print("[6] YOLO 훈련 실행...")
        rc = run_train_script(yolo_script, yaml_out_dir)
        if rc == 0:
            print("✅ 훈련 완료")
            rename_latest_yolo_run(yaml_out_dir, version_tag)
        else:
            print(f"❌ 오류 발생 (코드: {rc})")
    else:
        print("[!] 학습 실행을 건너뜀 (--skip-train)")


if __name__ == "__main__":
    main()
