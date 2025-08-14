#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 데이터 준비 · 학습 · 품질평가 · 결과 복사 파이프라인
DreamSim / LPIPS 양쪽 지원
"""

import argparse
import csv
import getpass
import json
import os
import shutil
import sys
import numpy as np
from pathlib import Path

# ── 내부 유틸 ───────────────────────────────────────────────────────────
from data_pipeline.utils_misc import (
    TeeWithTimestamp,
    copy_run_outputs,
    get_auto_log_filename,
    rename_latest_yolo_run,
)
from data_pipeline.dataset_config import CONFIGS
from data_pipeline.preprocess import add_prefix_to_filenames
from data_pipeline.config_utils import update_data_yaml_in_script
from data_pipeline.runner import run_train_script
from data_pipeline.quality_metrics import (
    compute_metric_per_image,  # LPIPS / DreamSim 공용
    compute_fid_per_versions,
    filter_dataset_by_metric,  # LPIPS 기반 함수 → 범용화 버전
)
# ───────────────────────────────────────────────────────────────────────


# ── 품질 평가 전용 ------------------------------------------------------

def run_quality_eval(dataset_root: str, *, versions: list[str], tag: str, metric: str):
    """FID + 선택 지표(LPIPS·DreamSim) 계산 & 로그 저장"""
    username = getpass.getuser()
    log_dir = Path(f"/home/{username}/jh_ws/yolo/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    train_dir = Path(dataset_root, "train")
    log_path  = log_dir / f"quality_{tag}.txt"

    # 1) FID -------------------------------------------------------------
    fid_tbl, skipped = compute_fid_per_versions(str(train_dir), versions)

    # 2) 모든 접두어 점수 ------------------------------------------------
    all_pairs = []
    for v in versions:                # v9, v10, v11, v12 …
        pairs, _ = compute_metric_per_image(
            str(train_dir), str(train_dir),
            prefix=f"{v}_", metric=metric
        )
        # 파일명을 'v9_base' 형태 그대로 저장
        all_pairs.extend([(f"{v}_{name}", score) for name, score in pairs])

    csv_path = Path(dataset_root, f"{metric}_scores.csv")
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows([("filename", metric), *all_pairs])

    # ── 통계 -----------------------------------------------------------
    scores = np.array([s for _, s in all_pairs])
    stats = {
        "mean":   float(scores.mean()),
        "median": float(np.median(scores)),
        "std":    float(scores.std()),
        "min":    float(scores.min()),
        "max":    float(scores.max()),
        "n_pairs": int(scores.size),
    }

    # 3) 로그 -----------------------------------------------------------
    with log_path.open("w") as f:
        for v, sc in fid_tbl.items():
            f.write(f"FID_{v}: {sc:.4f}\n")
        if skipped:
            skipped_str = ", ".join(f"{v}[orig={o},gen={g}]" for v, (o, g) in skipped.items())
            f.write(f"Skipped : {skipped_str}\n")
        f.write(f"{metric.upper()} stats:\n" + json.dumps(stats, indent=2) + "\n")
        f.write(f"{metric.upper()} CSV: {csv_path}\n")

    print(f"[INFO] 품질 결과 저장 → {log_path}")

# ── 메인 ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="여러 버전은 콤마/공백 모두 허용 (예: v9,v10,v11)")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--match-ratio", type=float, default=1.0)
    ap.add_argument(
        "--match-mode",
        choices=["mix", "vs"],
        default="mix",
        help="mix: 기존 섞기 / verselect: 버전 단위 전체 선택",
    )

    # ▶ 두 종류 시드 ------------------------------------------------------
    ap.add_argument("--seed-del", type=int)
    ap.add_argument("--seed-match", type=int)
    ap.add_argument("--seed", type=int, default=42, help="위 둘을 생략했을 때 쓰이는 공통 시드")

    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--save-results", action="store_true", help="학습 성공 시 results.csv·best.pt를 results/ 로 복사")

    # 빠른 모드 -----------------------------------------------------------
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--quality-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")

    # ── 품질 지표·필터 옵션 -------------------------------------------
    ap.add_argument("--metric", choices=["lpips", "dreamsim"], default="lpips", help="품질 지표 선택 (lpips | dreamsim)")
    ap.add_argument("--dataset", choices=["odsr", "tirod"], default="odsr", help="데이터셋 키 (odsr | tirod)")
    ap.add_argument("--sampling", choices=["random", "interval"], default="random", help="원본 삭제 샘플링 방식 (interval 은 tirod 전용)")

    ap.add_argument("--lpips-mode", choices=["range", "top", "bottom", "split"], default="range")
    ap.add_argument("--lpips-min", type=float)
    ap.add_argument("--lpips-max", type=float)
    ap.add_argument("--lpips-percent", type=float)
    ap.add_argument("--lpips-split", type=int, help="데이터를 k 등분할 때의 k 값")
    ap.add_argument("--lpips-split-idx", type=int, help="등분된 구간 중 사용할 인덱스(0-기반)")

    args = ap.parse_args()
    cfg = CONFIGS[args.dataset]

    username = getpass.getuser()
    yolo_base_path = f"/home/{username}/jh_ws/yolo"

    # ── analyze-only ---------------------------------------------------
    if args.analyze_only:
        root = yolo_base_path
        for d in os.listdir(Path(root, "runs/detect")):
            if d.startswith("train"):
                copy_run_outputs(root, d, results_dir="results")
        return

    if not args.version:
        ap.error("the following arguments are required: --version")

    versions = [v.strip() for v in args.version.replace(",", " ").split()]
    delete_ratio = args.ratio
    match_ratio = args.match_ratio
    sd = args.seed_del if args.seed_del is not None else args.seed
    sm = args.seed_match if args.seed_match is not None else args.seed

    tag = f"{'-'.join(versions)}_r{int(delete_ratio*100)}_m{int(match_ratio*100)}_sd{sd}_sm{sm}"

    base_dir = f"{yolo_base_path}/datasets"
    yolo_root = yolo_base_path
    yolo_script = cfg.default_yolo_script

    train_root = Path(base_dir, f"{cfg.name}_{tag}")
    train_dir = train_root / "train"

    # 로그 파일 중계 ------------------------------------------------------
    sys.stdout = TeeWithTimestamp(get_auto_log_filename(tag))
    sys.stderr = sys.stdout
    print(f"[INFO] TAG = {tag}")

    # ── quality-only ---------------------------------------------------
    if args.quality_only:
        run_quality_eval(str(train_root), versions=versions, tag=tag, metric=args.metric)
        return

    # ── 사전 접두어 만들기 --------------------------------------------
    if args.prefix:
        for v in versions:
            add_prefix_to_filenames(v, f"output/{cfg.output_prefix}_anno", f"output/{cfg.output_prefix}_{v}_anno")
            add_prefix_to_filenames(v, f"output/{cfg.output_prefix}", f"output/{cfg.output_prefix}_{v}")

    # ── 1) 데이터 복사 & 원본 삭제 ───────────────────────────────────
    if args.sampling == "interval" and not cfg.supports_interval:
        raise ValueError("interval 샘플링은 이 데이터셋에서 지원되지 않습니다.")

    if cfg.key == "odsr":
        # ODSR 시그니처: (base_dir, version, delete_ratio, dataset_name, seed, version_tag)
        cfg.copy_prune_fn(
            base_dir,
            versions[0],            # 첫 번째 버전
            delete_ratio,
            dataset_name=cfg.name,
            seed=sd,
            version_tag=tag,
        )
    else:  # tirod
        # TiROD 시그니처: (base_dir, ratio, seed, *, version_tag, sampling)
        cfg.copy_prune_fn(
            base_dir,
            delete_ratio,           # 삭제 비율
            seed=sd,
            version_tag=tag,
            sampling=args.sampling,
        )

    # ── 2) 접두어 매칭 복사 ─────────────────────────────────────────
    if args.match_mode == "mix":
        if match_ratio <= 0:
            print("[INFO] mix 모드: match_ratio=0 → 접두어 복사 생략 (원본만 학습)")
        else:
            # 기존 방식: 버전 전체를 하나로 합쳐서 무작위 샘플링
            cfg.copy_aug_fn(
                str(train_dir),
                versions,
                match_ratio=match_ratio,
                seed=sm,
            )
    else:  # vs (verselect)
        # 필수 검증 ----------------------------------------------------
        # → 0 이상의 정수 허용 (0이면 접두어 복사 생략 = 원본만 학습)
        if match_ratio < 0 or match_ratio != int(match_ratio):
            raise ValueError("--match-mode vs 에서는 --match-ratio 를 0 이상의 정수로 주세요.")
        k = int(match_ratio)
        
        if k > len(versions):
            raise ValueError(f"선택 버전 개수(k={k}) > 입력된 버전 수({len(versions)})")

        if k == 0:
            print("[INFO] vs 모드: match_ratio=0 → 접두어 복사 생략 (원본만 학습)")
            chosen = []
        else:
            # 재현성 있는 무작위 선택
            rng = np.random.default_rng(sm)
            chosen = rng.choice(versions, size=k, replace=False).tolist()
            print(f"[INFO] vs 모드: 선택된 버전 → {', '.join(chosen)}")

            # 선택된 각 버전을 **전부** 복사 (match_ratio=1.0)
            for v in chosen:
                cfg.copy_aug_fn(
                    str(train_dir),
                    [v],
                    match_ratio=1.0,
                    seed=sm,
                )

        # TAG 정보 확장
        # ── TAG 확장 & 데이터 폴더도 함께 이름 변경 ─────────────
        old_root = train_root                     # 현재 폴더
        tag     += f"_vs{k}"
        new_root = Path(base_dir, f"{cfg.name}_{tag}")
        if old_root.exists():
            old_root.rename(new_root)             # 폴더 개명
        train_root = new_root                     # 이후 경로 교체
        train_dir  = new_root / "train"

    # ── 3) 품질 평가 (필요 시) ---------------------------------------
    need_quality = (
        args.quality or args.lpips_mode in ("top", "bottom", "split") or args.lpips_min is not None
    )
    if need_quality:
        run_quality_eval(str(train_root), versions=versions, tag=tag, metric=args.metric)

    # ── 4) 지표 기반 필터 --------------------------------------------
    need_filter = (
        (args.lpips_mode == "range" and args.lpips_min is not None and args.lpips_max is not None)
        or (args.lpips_mode in ("top", "bottom") and args.lpips_percent is not None)
        or (
            args.lpips_mode == "split" and args.lpips_split is not None and args.lpips_split_idx is not None
        )
    )

    if need_filter:
        tmp_root, n_pref, n_orig, metric_stats = filter_dataset_by_metric(
            csv_path=train_root / f"{args.metric}_scores.csv",
            train_dir=str(train_dir),
            prefixes=versions,
            metric_name=args.metric,
            mode=args.lpips_mode,
            lpips_min=args.lpips_min,
            lpips_max=args.lpips_max,
            percent=args.lpips_percent,
            split_k=args.lpips_split,
            split_idx=args.lpips_split_idx,
        )

        # 후처리·이름 부여 --------------------------------------------
        if args.lpips_mode == "range":
            suf = f"_lp{args.lpips_min}-{args.lpips_max}"
        elif args.lpips_mode in ("top", "bottom"):
            suf = f"_{args.lpips_mode}{int(args.lpips_percent)}p"
        else:  # split
            suf = f"_split{args.lpips_split_idx}of{args.lpips_split}"

        tag += suf
        final_root = Path(base_dir, f"{cfg.name}_{tag}")
        if final_root.exists():
            shutil.rmtree(final_root)
        shutil.move(tmp_root, final_root)
        train_root, train_dir = final_root, final_root / "train"

        print(f"[INFO] 필터 완료 → {final_root} (orig={n_orig}, pref={n_pref})")

        # 통계 JSON ----------------------------------------------------
        json.dump(
            {
                "version_tag": tag,
                "orig_images": n_orig,
                "pref_images": n_pref,
                "total_images": n_orig + n_pref,
                "delete_ratio": delete_ratio,
                "match_ratio": match_ratio,
                "seed_del": sd,
                "seed_match": sm,
                "lpips_mode": args.lpips_mode,
                "lpips_min": args.lpips_min,
                "lpips_max": args.lpips_max,
                "lpips_percent": args.lpips_percent,
                "lpips_split": args.lpips_split,
                "lpips_split_idx": args.lpips_split_idx,
                "metric": args.metric,
                "metric_stats": metric_stats
            },
            open(final_root / "dataset_stats.json", "w"),
            indent=2,
        )

    # ── 5) YAML & 스크립트 -------------------------------------------
    cfg.create_yaml_fn(tag, save_path=yolo_root, base_dir=base_dir)
    update_data_yaml_in_script(yolo_script, f"{tag}.yaml")

    # ── 6) 학습 ------------------------------------------------------
    if args.skip_train:
        print("[!] --skip-train 지정, 학습 생략")
        return

    if run_train_script(yolo_script, yolo_root):
        print("❌ 학습 실패")
        return
    print("✅ 학습 완료")
    run_name = rename_latest_yolo_run(yolo_root, tag)
    if args.save_results and run_name:
        copy_run_outputs(yolo_root, run_name, results_dir="results")

if __name__ == "__main__":
    main()