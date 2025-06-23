#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 데이터 준비 · 학습 · 품질평가 · 결과 복사 파이프라인
"""
import argparse, os, sys, datetime, csv, json, shutil

# ─── 내부 유틸 ──────────────────────────────────────────────────────────
from data_pipeline.utils_misc import (
    TeeWithTimestamp, get_auto_log_filename,
    rename_latest_yolo_run, copy_run_outputs
)
from data_pipeline.preprocess import (
    add_prefix_to_filenames, copy_and_prune_dataset, copy_augmented_files
)
from data_pipeline.config_utils import create_yaml, update_data_yaml_in_script
from data_pipeline.runner import run_train_script
from data_pipeline.quality_metrics import compute_fid, compute_lpips_per_image, split_dataset_for_fid, compute_fid_per_versions, filter_dataset_by_lpips
# ───────────────────────────────────────────────────────────────────────

# 품질 평가(옵션) ──────────────────────────────────────────────────────
def run_quality_eval(dataset_root: str, versions: list[str], version_tag: str):
    """
    dataset_root = .../ODSR-IHS_<tag>
    - 버전별(prefix별) FID : 원본·접두어 짝이 맞는 이미지만 비교
    - LPIPS : 모든 짝을 개별 csv, 통계 요약
    결과:
      logs/quality_<tag>.txt
      results/lpips_<tag>.csv
    """
    train_dir = os.path.join(dataset_root, "train")
    log_dir   = "/home/jhcha/jh_ws/yolo/logs"
    res_dir   = "/home/jhcha/jh_ws/yolo/results"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    log_path  = os.path.join(log_dir, f"quality_{version_tag}.txt")
    csv_path  = os.path.join(dataset_root, "lpips_scores.csv")

    # ── 1) 버전별 FID ───────────────────────────────────────────────
    fid_table, skipped = compute_fid_per_versions(train_dir, versions)

    # ── 2) LPIPS (per-image) ───────────────────────────────────────
    prefix = f"{versions[0]}_"
    try:
        pairs, stats = compute_lpips_per_image(train_dir, train_dir, prefix=prefix)
        # CSV 저장
        with open(csv_path, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["filename", "lpips"])
            w.writerows(pairs)
    except Exception as e:
        stats, lpips_err = None, str(e)

    # ── 3) 로그 파일 작성 ──────────────────────────────────────────
    with open(log_path, "w") as flog:
        # FID
        for v, score in fid_table.items():
            flog.write(f"FID_{v}: {score:.4f}\n")
        if skipped:
            flog.write("Skipped (no matched pairs): " + ", ".join(
                f"{v}[orig={o},gen={g}]" for v, (o, g) in skipped.items()) + "\n")

        # LPIPS
        if stats:
            flog.write("LPIPS stats:\n" + json.dumps(stats, indent=2) + "\n")
            flog.write(f"LPIPS CSV: {csv_path}\n")
        else:
            flog.write(f"LPIPS 계산 실패: {lpips_err}\n")

    print(f"[INFO] 품질 결과 저장 → {log_path}")

# ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=False,
                        help="comma / space 로 여러 버전 지정. analyze-only·quality-only 모드에선 생략 가능")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--match-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--save-results", action="store_true",
                        help="훈련 성공 시 results.csv·best.pt 을 results/ 로 복사")
    parser.add_argument("--analyze-only", action="store_true",
                        help="runs/detect 의 모든 train* 폴더 결과만 복사 후 종료")
    parser.add_argument("--quality",      action="store_true",
                        help="파이프라인 실행 후 FID·LPIPS 평가 추가 실행")
    parser.add_argument("--quality-only", action="store_true",
                        help="데이터 준비·학습 없이 품질 평가만 실행")
    parser.add_argument("--lpips-mode", choices=["range", "top", "bottom"], default="range")
    parser.add_argument("--lpips-min", type=float)
    parser.add_argument("--lpips-max", type=float)
    parser.add_argument("--lpips-percent", type=float,
                    help="상·하위 모드에서 사용할 퍼센트 (예 10 → 10%)")
    args = parser.parse_args()

    # ---------- 2. analyze-only ----------
    if args.analyze_only:
        yolo_root = "/home/jhcha/jh_ws/yolo"
        for d in os.listdir(os.path.join(yolo_root, "runs", "detect")):
            if d.startswith("train"):
                copy_run_outputs(yolo_root, d, results_dir="results")
        return

    # ---------- 3. version 필수 여부 ----------
    if (args.quality_only or args.lpips_mode or args.lpips_min or
        args.lpips_max or args.lpips_percent) and not args.version:
        parser.error("--version 을 지정해야 합니다.")

    if not args.version and not args.analyze_only:
        parser.error("--version 인자는 필수입니다.")

    # ---------- 4. 태그/경로 공통 ----------
    versions     = [v.strip() for v in args.version.replace(",", " ").split()]
    delete_ratio = args.ratio
    match_ratio  = args.match_ratio
    version_tag  = f"{'-'.join(versions)}_r{int(delete_ratio*100)}_m{int(match_ratio*100)}"
    base_dir     = "/home/jhcha/jh_ws/yolo/datasets"
    yolo_root    = "/home/jhcha/jh_ws/yolo"
    yolo_script  = "/home/jhcha/jh_ws/yolo/yolo_train_ODSR_half.py"
    train_root   = os.path.join(base_dir, f"ODSR-IHS_{version_tag}")
    train_dir    = os.path.join(train_root, "train")

    # ---------- 5. 로그 ----------
    log_file = get_auto_log_filename(version_tag)
    sys.stdout = TeeWithTimestamp(log_file); sys.stderr = sys.stdout
    print(f"[INFO] 로그 파일: {log_file}")

    # ---------- 6. quality-only (기존 데이터 평가만) ----------
    if args.quality_only:
        run_quality_eval(train_root, versions, version_tag)
        return

    # ---------- 7. 데이터 준비 ----------
    if args.prefix:
        for v in versions:
            add_prefix_to_filenames(v, "output/ODSR_anno", f"output/ODSR_{v}_anno")
            add_prefix_to_filenames(v, "output/ODSR",      f"output/ODSR_{v}")

    copy_and_prune_dataset(base_dir, versions[0], delete_ratio,
                           seed=args.seed, version_tag=version_tag)

    copy_augmented_files(train_dir, versions,
                         match_ratio=match_ratio, seed=args.seed)

    # ---------- 8. LPIPS/FID 계산 (무조건 필요: 필터 or --quality) ----------
    need_quality = args.quality or args.lpips_min is not None or args.lpips_percent is not None
    if need_quality:
        run_quality_eval(train_root, versions, version_tag)

    # ---------- 9. LPIPS 기반 필터링 -----------------------------------
    need_filter = (
        (args.lpips_mode == "range" and args.lpips_min is not None and args.lpips_max is not None) or
        (args.lpips_mode in ("top", "bottom") and args.lpips_percent is not None)
    )

    base_tag = version_tag

    if need_filter:
        # 9-1. CSV 경로 (quality_eval 에서 저장한 파일)
        csv_path = os.path.join(train_root, "lpips_scores.csv")
        prefix   = f"{versions[0]}_"

    # 9-2. 임시 폴더에 필터링된 데이터셋 만들기
    tmp_root, n_pref, n_orig = filter_dataset_by_lpips(
        csv_path=csv_path,
        train_dir=train_dir,
        prefix=prefix,
        mode=args.lpips_mode,
        lpips_min=args.lpips_min,
        lpips_max=args.lpips_max,
        percent=args.lpips_percent,
    )

    # 9-3. datasets/ODSR-IHS_<tag+suffix> 로 이동
    filt_tag = (
        f"_lp{args.lpips_min}-{args.lpips_max}"
        if args.lpips_mode == "range"
        else f"_{args.lpips_mode}{int(args.lpips_percent)}p"
    )
    version_tag += filt_tag
    final_root = os.path.join(base_dir, f"ODSR-IHS_{version_tag}")

    if os.path.exists(final_root):
        shutil.rmtree(final_root)
    shutil.move(tmp_root, final_root)

    # 9-4. 경로 갱신
    train_root = final_root
    train_dir  = os.path.join(final_root, "train")
    

    print(f"[INFO] LPIPS 필터링 완료 → {final_root} "
        f"(orig_images={n_orig}, pref_images={n_pref})")
    
    total_after = n_pref + n_orig

        # ── (A) quality 로그(txt)에 추가 ─────────────────────
    qual_log = os.path.join("/home/jhcha/jh_ws/yolo/logs",
                            f"quality_{base_tag}.txt")
    with open(qual_log, "a") as f:
        f.write(f"Dataset counts  |  orig={n_orig}, pref={n_pref}, total={total_after}\n")

    # ── (B) 구조화 JSON 저장 ────────────────────────────
    stats_json = {
        "version_tag": version_tag,
        "orig_images": n_orig,
        "pref_images": n_pref,
        "total_images": total_after,
        "lpips_mode": args.lpips_mode,
        "lpips_min": args.lpips_min,
        "lpips_max": args.lpips_max,
        "lpips_percent": args.lpips_percent,
        "delete_ratio": delete_ratio,
        "match_ratio": match_ratio
    }
    with open(os.path.join(final_root, "dataset_stats.json"), "w") as fj:
        json.dump(stats_json, fj, indent=2)
    print(f"[INFO] 통계 JSON 저장 → {final_root}/dataset_stats.json")
        

    # ----------10. YAML 생성 & 스크립트 수정 ----------
    create_yaml(version_tag, delete_ratio, match_ratio,
                save_path=yolo_root, base_dir=base_dir)
    update_data_yaml_in_script(yolo_script, f"{version_tag}.yaml")

    # ----------11. 학습 ----------
    if args.skip_train:
        print("[!] --skip-train : 학습 건너뜀")
        return

    rc = run_train_script(yolo_script, yolo_root)
    if rc != 0:
        print(f"❌ 훈련 실패(code {rc})"); return
    print("✅ 훈련 완료")

    run_name = rename_latest_yolo_run(yolo_root, version_tag)
    if args.save_results and run_name:
        copy_run_outputs(yolo_root, run_name, results_dir="results")
if __name__ == "__main__":
    main()
