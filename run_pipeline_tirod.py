#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TiROD 데이터셋용 YOLO 전처리·학습·품질평가 파이프라인
=====================================================
ODSR 기반 `run_pipeline.py`를 그대로 계승하되 **데이터셋 이름만 TiROD로 교체**했습니다.
사용 방법, CLI 옵션, 내부 로그 구조는 기존 스크립트와 100% 동일합니다.
(단, YAML 생성 함수와 기본 YOLO 학습 스크립트 경로가 달라졌습니다.)

주요 변경점
------------
* 데이터셋 원본 폴더:  ``datasets/TiROD``
* 증강본/어노테이션 위치: ``output/TiROD_<ver>``, ``output/TiROD_<ver>_anno``
* 새 데이터셋 복사본:  ``datasets/TiROD_<tag>``
* YAML 생성 함수   :  ``create_tirod_yaml`` (config_utils.py)
* 기본 학습 스크립트:  ``yolo_train_TiROD.py`` (경로는 필요 시 수정)

나머지 로직·옵션은 ODSR 파이프라인과 동일합니다.  
필요하면 `--help` 로 CLI 인자를 확인하세요.
"""

from __future__ import annotations

import argparse, os, sys, json, csv, shutil, getpass, random, tempfile
from pathlib import Path

# ── 공통 유틸 ──────────────────────────────────────────────────────────
from data_pipeline.utils_misc import (
    TeeWithTimestamp, get_auto_log_filename,
    rename_latest_yolo_run, copy_run_outputs,
)
from data_pipeline.preprocess import (
    add_prefix_to_filenames,  # 그대로 재사용
)
from data_pipeline.config_utils import (
    create_tirod_yaml, update_data_yaml_in_script,
)
from data_pipeline.runner import run_train_script
from data_pipeline.quality_metrics import (
    compute_fid_per_versions, compute_lpips_per_image, filter_dataset_by_lpips,
)

# ── 상수 ──────────────────────────────────────────────────────────────
DATASET_NAME = "TiROD"          # output / datasets 폴더명용 프리픽스
VALID_EXT = {".jpg", ".jpeg", ".png"}

# ────────────────────────── 데이터 복사/삭제 ───────────────────────────

def copy_and_prune_dataset_tirod(
    base_dir: str,
    ratio: float,
    seed: int = 42,
    *,
    version_tag: str,
):
    """datasets/TiROD → datasets/TiROD_<tag> 복사 후 `ratio` 만큼 train 이미지 무작위 삭제"""
    src = os.path.join(base_dir, DATASET_NAME)
    dst = os.path.join(base_dir, f"{DATASET_NAME}_{version_tag}")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    train_dir = os.path.join(dst, "train")
    imgs = [f for f in os.listdir(train_dir) if Path(f).suffix.lower() in VALID_EXT]
    random.seed(seed)
    del_cnt = int(len(imgs) * ratio)

    for f in random.sample(imgs, del_cnt):
        os.remove(os.path.join(train_dir, f))
        txt = os.path.join(train_dir, Path(f).with_suffix(".txt").name)
        if os.path.exists(txt):
            os.remove(txt)

    print(f"'{src}' → '{dst}' 복사 및 {del_cnt}개 파일 삭제 완료")

# ──────────────────────── 증강본 매칭 복사 ─────────────────────────────

def _copy_one(src: Path, dst_dir: Path):
    dst = dst_dir / src.name
    shutil.copy(src, dst)


def copy_augmented_files_tirod(
    train_dir: str,
    versions: list[str] | str,
    *,
    match_ratio: float = 1.0,
    base_output: str = "output",
    seed: int = 42,
):
    """ODSR 버전 함수와 동일 로직이지만 폴더 프리픽스를 TiROD 로 바꾼 구현"""
    if isinstance(versions, str):
        versions = [versions]

    random.seed(seed)
    train_dir_p = Path(train_dir)

    # ── (0) 접두어 없는 원본 base 이름 수집 ────────────────────────────
    base_names = [
        f.stem
        for f in train_dir_p.iterdir()
        if f.suffix.lower() in VALID_EXT and not any(f.name.startswith(f"{v}_") for v in versions)
    ]

    # 단일 버전 or match_ratio < 1.0 -----------------------------------
    if len(versions) == 1 or match_ratio < 1.0:
        v = versions[0]
        img_src_dir = Path(base_output) / f"{DATASET_NAME}_{v}"
        lbl_src_dir = Path(base_output) / f"{DATASET_NAME}_{v}_anno"
        pref = f"{v}_"

        valid = [
            n for n in base_names
            if any((img_src_dir / f"{pref}{n}{ext}").exists() for ext in VALID_EXT)
            and (lbl_src_dir / f"{pref}{n}.txt").exists()
        ]
        k = int(len(valid) * match_ratio)
        chosen = random.sample(valid, k)

        for n in chosen:
            for ext in VALID_EXT:
                fp = img_src_dir / f"{pref}{n}{ext}"
                if fp.exists():
                    _copy_one(fp, train_dir_p)
                    break
            _copy_one(lbl_src_dir / f"{pref}{n}.txt", train_dir_p)

        print(f"[INFO] 접두어 복사 {k}쌍 ({v}, match_ratio={match_ratio})")
        return

    # 다중 버전 & match_ratio ≥ 1.0 ------------------------------------
    candidates: list[tuple[str, str]] = []      # (base_name, version)

    for v in versions:
        img_dir = Path(base_output) / f"{DATASET_NAME}_{v}"
        lbl_dir = Path(base_output) / f"{DATASET_NAME}_{v}_anno"
        pref = f"{v}_"

        for n in base_names:
            if (
                any((img_dir / f"{pref}{n}{ext}").exists() for ext in VALID_EXT)
                and (lbl_dir / f"{pref}{n}.txt").exists()
            ):
                candidates.append((n, v))

    if not candidates:
        print("[WARN] 매칭 가능한 접두어 파일이 없습니다.")
        return

    target = min(int(len(base_names) * match_ratio), len(candidates))
    picked = random.sample(candidates, target)

    for n, v in picked:
        img_dir = Path(base_output) / f"{DATASET_NAME}_{v}"
        lbl_dir = Path(base_output) / f"{DATASET_NAME}_{v}_anno"
        pref = f"{v}_"
        for ext in VALID_EXT:
            fp = img_dir / f"{pref}{n}{ext}"
            if fp.exists():
                _copy_one(fp, train_dir_p)
                break
        _copy_one(lbl_dir / f"{pref}{n}.txt", train_dir_p)

    actual = target / max(1, len(base_names))
    sat = " (saturated)" if target < int(len(base_names) * match_ratio) else ""
    print(
        f"[INFO] 접두어 복사 {target}쌍 / {len(base_names)} (actual match-ratio≈{actual:.2f}){sat}"
    )

# ─────────────────────────── 품질 평가 래퍼 ────────────────────────────

def run_quality_eval(dataset_root: str, versions: list[str], tag: str):
    username = getpass.getuser()
    log_dir = f"/home/{username}/jh_ws/yolo/logs"
    os.makedirs(log_dir, exist_ok=True)

    train_dir = Path(dataset_root, "train")
    log_path = Path(log_dir, f"quality_{tag}.txt")
    csv_path = Path(dataset_root, "lpips_scores.csv")

    # 1) FID (버전별)
    fid_tbl, skipped = compute_fid_per_versions(str(train_dir), versions)

    # 2) LPIPS (첫 번째 버전 기준)
    pairs, stats = compute_lpips_per_image(
        str(train_dir), str(train_dir), prefix=f"{versions[0]}_"
    )
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerows([("filename", "lpips"), *pairs])

    # 3) 로그 정리
    with log_path.open("w") as f:
        for v, sc in fid_tbl.items():
            f.write(f"FID_{v}: {sc:.4f}\n")
        if skipped:
            f.write(
                "Skipped : "
                + ", ".join(f"{v}[orig={o},gen={g}]" for v, (o, g) in skipped.items())
                + "\n"
            )
        f.write("LPIPS stats:\n" + json.dumps(stats, indent=2) + "\n")
        f.write(f"LPIPS CSV: {csv_path}\n")

    print(f"[INFO] 품질 결과 저장 → {log_path}")

# ──────────────────────────────── 메인 ────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="여러 버전은 콤마/공백 모두 허용 (예: v1,v2,v3)")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--match-ratio", type=float, default=1.0)

    # ▶ 두 종류 시드
    ap.add_argument("--seed-del", type=int)
    ap.add_argument("--seed-match", type=int)
    ap.add_argument("--seed", type=int, default=42, help="위 둘을 생략했을 때 쓰이는 공통 시드")

    ap.add_argument("--prefix", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--save-results", action="store_true", help="학습 성공 시 results.csv·best.pt를 results/ 로 복사")

    # 빠른 모드
    ap.add_argument("--quality", action="store_true")
    ap.add_argument("--quality-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")

    # ── LPIPS 필터 옵션 ────────────────────────────────────────────
    ap.add_argument("--lpips-mode", choices=["range", "top", "bottom", "split"], default="range")
    ap.add_argument("--lpips-min", type=float)
    ap.add_argument("--lpips-max", type=float)
    ap.add_argument("--lpips-percent", type=float)
    ap.add_argument("--lpips-split", type=int, help="데이터를 k 등분할 때의 k 값")
    ap.add_argument("--lpips-split-idx", type=int, help="등분된 구간 중 사용할 인덱스(0-기반)")

    args = ap.parse_args()

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

    tag = f"{'-'.join(versions)}_r{int(delete_ratio * 100)}_m{int(match_ratio * 100)}_sd{sd}_sm{sm}"

    base_dir = f"{yolo_base_path}/datasets"
    yolo_root = yolo_base_path
    yolo_script = f"{yolo_base_path}/yolo_train_TiROD.py"  # 사용자 교체 가능

    train_root = Path(base_dir, f"{DATASET_NAME}_{tag}")
    train_dir = train_root / "train"

    # 로그 파일 중계
    sys.stdout = TeeWithTimestamp(get_auto_log_filename(tag))
    sys.stderr = sys.stdout
    print(f"[INFO] TAG = {tag}")

    # ── quality-only ---------------------------------------------------
    if args.quality_only:
        run_quality_eval(str(train_root), versions, tag)
        return

    # ── 사전 접두어 만들기(옵션) ----------------------------------------
    if args.prefix:
        for v in versions:
            add_prefix_to_filenames(v, f"output/{DATASET_NAME}_anno", f"output/{DATASET_NAME}_{v}_anno")
            add_prefix_to_filenames(v, f"output/{DATASET_NAME}", f"output/{DATASET_NAME}_{v}")

    # ── 1) 데이터 복사 & 원본 삭제 --------------------------------------
    copy_and_prune_dataset_tirod(
        base_dir, delete_ratio, seed=sd, version_tag=tag
    )

    # ── 2) 접두어 매칭 복사 -------------------------------------------
    copy_augmented_files_tirod(
        str(train_dir), versions, match_ratio=match_ratio, seed=sm
    )

    # ── 3) 품질 평가 (필요 시) -----------------------------------------
    need_quality = (
        args.quality
        or args.lpips_mode in ("top", "bottom", "split")
        or args.lpips_min is not None
    )
    if need_quality:
        run_quality_eval(str(train_root), versions, tag)

    # ── 4) LPIPS 필터 --------------------------------------------------
    need_filter = (
        (args.lpips_mode == "range" and args.lpips_min is not None and args.lpips_max is not None)
        or (args.lpips_mode in ("top", "bottom") and args.lpips_percent is not None)
        or (
            args.lpips_mode == "split" and args.lpips_split is not None and args.lpips_split_idx is not None
        )
    )

    if need_filter:
        tmp_root, n_pref, n_orig = filter_dataset_by_lpips(
            csv_path=train_root / "lpips_scores.csv",
            train_dir=str(train_dir),
            prefix=f"{versions[0]}_",
            mode=args.lpips_mode,
            lpips_min=args.lpips_min,
            lpips_max=args.lpips_max,
            percent=args.lpips_percent,
            split_k=args.lpips_split,
            split_idx=args.lpips_split_idx,
        )

        # 후처리·이름 부여
        if args.lpips_mode == "range":
            suf = f"_lp{args.lpips_min}-{args.lpips_max}"
        elif args.lpips_mode in ("top", "bottom"):
            suf = f"_{args.lpips_mode}{int(args.lpips_percent)}p"
        else:  # split
            suf = f"_split{args.lpips_split_idx}of{args.lpips_split}"

        tag += suf
        final_root = Path(base_dir, f"{DATASET_NAME}_{tag}")
        if final_root.exists():
            shutil.rmtree(final_root)
        shutil.move(tmp_root, final_root)
        train_root, train_dir = final_root, final_root / "train"

        print(
            f"[INFO] LPIPS 필터 완료 → {final_root} (orig={n_orig}, pref={n_pref})"
        )

        # 통계 JSON 저장
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
            },
            open(final_root / "dataset_stats.json", "w"),
            indent=2,
        )

    # ── 5) YAML & 스크립트 --------------------------------------------
    create_tirod_yaml(                       # ← TiROD 전용 YAML 생성 진입점
    tag,
    save_path=yolo_root,
    base_dir=base_dir,
    )
    update_data_yaml_in_script(
        yolo_script,                        # yolo_train_TiROD.py
        f"{tag}.yaml",
    )

    # ── 6) 학습 --------------------------------------------------------
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
