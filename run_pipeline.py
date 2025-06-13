#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 데이터 준비·학습·결과 복사 파이프라인
"""
import argparse, os, sys, datetime
from data_pipeline.utils_misc import (
    TeeWithTimestamp, get_auto_log_filename,
    rename_latest_yolo_run, copy_run_outputs
)
from data_pipeline.preprocess import (
    add_prefix_to_filenames, copy_and_prune_dataset, copy_augmented_files
)
from data_pipeline.config_utils import create_yaml, update_data_yaml_in_script
from data_pipeline.runner import run_train_script

# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=False,
                        help="Not necessary on analyze-only mode")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--match-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--save-results", action="store_true",
                        help="훈련 성공 시 결과(csv, pt)를 results/ 로 복사")
    parser.add_argument("--analyze-only", action="store_true",
                        help="runs/detect 내 모든 train* 폴더 결과만 복사하고 종료")
    args = parser.parse_args()

    # ───────── analyze-only 모드 ─────────
    if args.analyze_only:
        yolo_root = "/home/jhcha2/jh_ws/yolo"
        detect_dir = os.path.join(yolo_root, "runs", "detect")
        for d in os.listdir(detect_dir):
            if d.startswith("train"):
                copy_run_outputs(yolo_root, d, results_dir="results")
        return
    
    if not args.analyze_only and not args.version:
        parser.error("--version 인자는 학습·전처리 모드에서 필수입니다.")

    # ───────── 학습 파이프라인 ─────────
    versions = [v.strip() for v in args.version.replace(",", " ").split() if v.strip()]
    delete_ratio, match_ratio, random_seed = args.ratio, args.match_ratio, args.seed
    version_tag = f"{'-'.join(versions)}_r{int(delete_ratio*100)}_m{int(match_ratio*100)}"

    # 로그
    log_file = get_auto_log_filename(version_tag)
    sys.stdout = TeeWithTimestamp(log_file); sys.stderr = sys.stdout
    print(f"[INFO] 로그: {log_file}")

    # 경로
    base_dir  = "/home/jhcha2/jh_ws/yolo/datasets"
    yolo_root = "/home/jhcha2/jh_ws/yolo"
    yolo_script = "/home/jhcha2/jh_ws/yolo/yolo_train_ODSR_half.py"
    train_dir = os.path.join(base_dir, f"ODSR-IHS_{version_tag}", "train")

    # 1. 접두어
    if args.prefix:
        for v in versions:
            add_prefix_to_filenames(v, "output/ODSR_anno", f"output/ODSR_{v}_anno")
            add_prefix_to_filenames(v, "output/ODSR",      f"output/ODSR_{v}")

    # 2. 복사·삭제
    copy_and_prune_dataset(base_dir, versions[0], delete_ratio,
                           seed=random_seed, version_tag=version_tag)

    # 3. 접두어 복원
    copy_augmented_files(train_dir, versions,
                         match_ratio=match_ratio, seed=random_seed)

    # 4. YAML
    create_yaml(version_tag, delete_ratio, match_ratio,
                save_path=yolo_root, base_dir=base_dir)

    # 5. 스크립트 수정
    update_data_yaml_in_script(yolo_script, f"{version_tag}.yaml")

    # 6. 훈련
    if args.skip_train:
        print("[!] --skip-train, 학습 생략"); return

    rc = run_train_script(yolo_script, yolo_root)
    if rc != 0:
        print(f"❌ 훈련 실패 (code {rc})"); return

    print("✅ 훈련 완료")
    run_name = rename_latest_yolo_run(yolo_root, version_tag)
    if args.save_results and run_name:
        copy_run_outputs(yolo_root, run_name, results_dir="results")


if __name__ == "__main__":
    main()
