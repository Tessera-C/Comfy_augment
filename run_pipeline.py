#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 데이터 준비 · 학습 · 품질평가 · 결과 복사 파이프라인
"""
import argparse, os, sys, datetime, csv, json, shutil

# ─── 내부 유틸 ───────────────────────────────────────────────────────────
from data_pipeline.utils_misc import (
    TeeWithTimestamp, get_auto_log_filename,
    rename_latest_yolo_run, copy_run_outputs,
)
from data_pipeline.preprocess import (
    add_prefix_to_filenames,
    copy_and_prune_dataset,
    copy_augmented_files,
)
from data_pipeline.config_utils import create_yaml, update_data_yaml_in_script
from data_pipeline.runner import run_train_script
from data_pipeline.quality_metrics import (
    compute_fid_per_versions,
    compute_lpips_per_image,
    filter_dataset_by_lpips,
)
# ────────────────────────────────────────────────────────────────────────


# ─── 품질 평가 전용 -----------------------------------------------------
def run_quality_eval(dataset_root: str, versions: list[str], version_tag: str):
    """FID(버전별)+LPIPS 계산 후 로그/CSV 저장"""
    train_dir = os.path.join(dataset_root, "train")
    log_dir   = "/home/jhcha/jh_ws/yolo/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_path  = os.path.join(log_dir, f"quality_{version_tag}.txt")
    csv_path  = os.path.join(dataset_root, "lpips_scores.csv")

    # 1) FID (버전별 원본–접두어 1:1 매칭)
    fid_table, skipped = compute_fid_per_versions(train_dir, versions)

    # 2) LPIPS (모든 짝)
    pairs, lpips_stats = compute_lpips_per_image(
        train_dir, train_dir, prefix=f"{versions[0]}_"
    )
    with open(csv_path, "w", newline="") as fcsv:
        csv.writer(fcsv).writerows([("filename", "lpips"), *pairs])

    # 3) 로그
    with open(log_path, "w") as f:
        for v, sc in fid_table.items():
            f.write(f"FID_{v}: {sc:.4f}\n")
        if skipped:
            f.write(
                "Skipped (no matched pairs): "
                + ", ".join(f"{v}[orig={o},gen={g}]" for v, (o, g) in skipped.items())
                + "\n"
            )
        f.write("LPIPS stats:\n" + json.dumps(lpips_stats, indent=2) + "\n")
        f.write(f"LPIPS CSV: {csv_path}\n")

    print(f"[INFO] 품질 결과 저장 → {log_path}")


# ─── 메인 ---------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=False,
                   help="v9,v10 … 공백/콤마 모두 허용")
    p.add_argument("--ratio",        type=float, default=0.5)
    p.add_argument("--match-ratio",  type=float, default=1.0)

    # ▶▶ 시드 2종 ◀◀
    p.add_argument("--seed-del",   type=int, help="원본 삭제 시드")
    p.add_argument("--seed-match", type=int, help="접두어 매칭 시드")
    p.add_argument("--seed",       type=int, default=42,
                   help="위 2개를 생략했을 때 함께 쓰이는 기본 시드")

    p.add_argument("--prefix", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--save-results", action="store_true")

    # 품질 옵션
    p.add_argument("--quality",      action="store_true")
    p.add_argument("--quality-only", action="store_true")
    p.add_argument("--analyze-only", action="store_true")

    # LPIPS 필터
    p.add_argument("--lpips-mode", choices=["range", "top", "bottom"], default="range")
    p.add_argument("--lpips-min",  type=float)
    p.add_argument("--lpips-max",  type=float)
    p.add_argument("--lpips-percent", type=float)

    args = p.parse_args()

    # ── 빠른 모드들 -----------------------------------------------------
    if args.analyze_only:
        yolo_root = "/home/jhcha/jh_ws/yolo"
        for d in os.listdir(os.path.join(yolo_root, "runs", "detect")):
            if d.startswith("train"):
                copy_run_outputs(yolo_root, d, results_dir="results")
        return
    if args.quality_only and not args.version:
        p.error("--quality-only 는 --version 필수")

    if not args.version:
        p.error("--version 인자는 필수입니다.")

    # ── 공통 파라미터 ---------------------------------------------------
    versions      = [v.strip() for v in args.version.replace(",", " ").split()]
    delete_ratio  = args.ratio
    match_ratio   = args.match_ratio
    seed_del      = args.seed_del   if args.seed_del   is not None else args.seed
    seed_match    = args.seed_match if args.seed_match is not None else args.seed

    version_tag = (
        f"{'-'.join(versions)}"
        f"_r{int(delete_ratio*100)}"
        f"_m{int(match_ratio*100)}"
        f"_sd{seed_del}_sm{seed_match}"
    )

    base_dir   = "/home/jhcha/jh_ws/yolo/datasets"
    yolo_root  = "/home/jhcha/jh_ws/yolo"
    yolo_script = "/home/jhcha/jh_ws/yolo/yolo_train_ODSR_half.py"
    train_root = os.path.join(base_dir, f"ODSR-IHS_{version_tag}")
    train_dir  = os.path.join(train_root, "train")

    # ── 로그 초기화 -----------------------------------------------------
    sys.stdout = TeeWithTimestamp(get_auto_log_filename(version_tag))
    sys.stderr = sys.stdout
    print(f"[INFO] TAG = {version_tag}")

    # ── quality-only ---------------------------------------------------
    if args.quality_only:
        run_quality_eval(train_root, versions, version_tag)
        return

    # ── 1) 접두어 생성(옵션) -------------------------------------------
    if args.prefix:
        for v in versions:
            add_prefix_to_filenames(v, "output/ODSR_anno", f"output/ODSR_{v}_anno")
            add_prefix_to_filenames(v, "output/ODSR",      f"output/ODSR_{v}")

    # ── 2) 원본 복사+삭제 ---------------------------------------------
    copy_and_prune_dataset(
        base_dir, versions[0], delete_ratio,
        seed=seed_del, version_tag=version_tag,
    )

    # ── 3) 접두어 매칭 복사 -------------------------------------------
    copy_augmented_files(
        train_dir, versions,
        match_ratio=match_ratio, seed=seed_match,
    )

    # ── 4) (선택) 품질 평가 ------------------------------------------
    need_quality = (
        args.quality
        or args.lpips_mode in ("top", "bottom")
        or args.lpips_min is not None
    )
    if need_quality:
        run_quality_eval(train_root, versions, version_tag)

    # ── 5) LPIPS 필터 --------------------------------------------------
    need_filter = (
        args.lpips_mode == "range" and args.lpips_min is not None and args.lpips_max is not None
    ) or (
        args.lpips_mode in ("top", "bottom") and args.lpips_percent is not None
    )

    if need_filter:
        csv_path = os.path.join(train_root, "lpips_scores.csv")
        tmp_root, n_pref, n_orig = filter_dataset_by_lpips(
            csv_path=csv_path,
            train_dir=train_dir,
            prefix=f"{versions[0]}_",
            mode=args.lpips_mode,
            lpips_min=args.lpips_min,
            lpips_max=args.lpips_max,
            percent=args.lpips_percent,
        )
        # suffix
        suf = (
            f"_lp{args.lpips_min}-{args.lpips_max}"
            if args.lpips_mode == "range" else
            f"_{args.lpips_mode}{int(args.lpips_percent)}p"
        )
        version_tag += suf
        final_root = os.path.join(base_dir, f"ODSR-IHS_{version_tag}")
        if os.path.exists(final_root):
            shutil.rmtree(final_root)
        shutil.move(tmp_root, final_root)
        train_root, train_dir = final_root, os.path.join(final_root, "train")
        print(f"[INFO] LPIPS 필터링 완료 → {final_root} "
              f"(orig={n_orig}, pref={n_pref})")

        # stats JSON
        json.dump(
            {
                "version_tag": version_tag,
                "orig_images": n_orig,
                "pref_images": n_pref,
                "total_images": n_orig + n_pref,
                "delete_ratio": delete_ratio,
                "match_ratio": match_ratio,
                "seed_del": seed_del,
                "seed_match": seed_match,
                "lpips_mode": args.lpips_mode,
                "lpips_min": args.lpips_min,
                "lpips_max": args.lpips_max,
                "lpips_percent": args.lpips_percent,
            },
            open(os.path.join(final_root, "dataset_stats.json"), "w"),
            indent=2,
        )

    # ── 6) YAML & 스크립트 --------------------------------------------
    create_yaml(version_tag, delete_ratio, match_ratio,
                save_path=yolo_root, base_dir=base_dir)
    update_data_yaml_in_script(yolo_script, f"{version_tag}.yaml")

    # ── 7) 학습 --------------------------------------------------------
    if args.skip_train:
        print("[!] --skip-train 지정, 학습 생략")
        return

    if run_train_script(yolo_script, yolo_root):
        print("❌ 학습 실패"); return
    print("✅ 학습 완료")

    run_name = rename_latest_yolo_run(yolo_root, version_tag)
    if args.save_results and run_name:
        copy_run_outputs(yolo_root, run_name, results_dir="results")


if __name__ == "__main__":
    main()
