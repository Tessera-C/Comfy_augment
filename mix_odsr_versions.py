#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ODSR 버전 믹서 (파일명 접두사도 대상 버전에 맞춰 교체)
- 입력: 여러 원본 버전(예: v9~v20)
- 출력: 동일 개수의 새 버전(예: v100~v112)
- 로직: 프레임 단위로 원본 버전들을 랜덤 순열로 섞어 새 버전에 1:1 배치
- 파일명: 대상 폴더 버전에 맞춰 선두 'v{num}_' 접두사를 'v{dest}_'로 교체 (라벨 포함)
"""

import argparse, re, shutil, random, sys, csv, os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

LEADING_VER_PREFIX_RE = re.compile(r"^v\d+_")

def parse_args():
    p = argparse.ArgumentParser(description="ODSR 버전 믹서")
    p.add_argument("--root", default="output", help="루트 폴더 (기본: output)")
    p.add_argument("--src-versions", required=True,
                   help="원본 버전 번호들, 예: 9,10,11,12")
    p.add_argument("--dest-start", type=int, required=True,
                   help="새 버전 시작 번호, 예: 100 (원본 개수만큼 연속 생성)")
    p.add_argument("--exts", default="png,jpg,jpeg",
                   help="이미지 확장자들(콤마 구분, 소문자) (기본: png,jpg,jpeg)")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    p.add_argument("--link", choices=["copy","hardlink","symlink","move"], default="copy",
                   help="배치 방식: copy(기본)/hardlink/symlink/move")
    p.add_argument("--strict", action="store_true",
                   help="모든 프레임이 모든 원본 버전에 존재하지 않으면 에러")
    p.add_argument("--overwrite", action="store_true",
                   help="기존 대상 폴더가 있으면 비우고 진행")
    p.add_argument("--csv-out", default=None,
                   help="배치 결과 매핑을 CSV로 저장 (선택)")
    p.add_argument("--keep-original-filenames", action="store_true",
                   help="파일명을 바꾸지 않고 원본 그대로 사용(라벨 포함)")
    return p.parse_args()

def ensure_clean_dir(d: Path, overwrite: bool):
    if d.exists():
        if not overwrite:
            print(f"[ERROR] 대상 디렉토리가 이미 존재합니다: {d}  (--overwrite 사용 가능)", file=sys.stderr)
            sys.exit(1)
        for p in d.rglob("*"):
            if p.is_file() or p.is_symlink():
                p.unlink()
    else:
        d.mkdir(parents=True, exist_ok=True)

def list_images(img_dir: Path, exts):
    files = []
    for ext in exts:
        files.extend(img_dir.glob(f"*.{ext}"))
        files.extend(img_dir.glob(f"*.{ext.upper()}"))
    return files

def frame_key_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    stem_wo_ver = LEADING_VER_PREFIX_RE.sub("", stem)
    return stem_wo_ver  # 예: '000003_jpg.rf.xxxxx'

def copy_like(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    elif mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unknown mode: {mode}")

def main():
    args = parse_args()
    random.seed(args.seed)
    root = Path(args.root)
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    src_versions = sorted([int(v.strip()) for v in args.src_versions.split(",")])
    K = len(src_versions)
    dest_versions = [args.dest_start + i for i in range(K)]

    # 원본 디렉토리 확인
    src_img_dirs = {}
    src_lbl_dirs = {}
    for v in src_versions:
        img_dir = root / f"ODSR_v{v}"
        lbl_dir = root / f"ODSR_v{v}_anno"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            print(f"[ERROR] 원본 디렉토리 누락: {img_dir} 또는 {lbl_dir}", file=sys.stderr)
            sys.exit(1)
        src_img_dirs[v] = img_dir
        src_lbl_dirs[v] = lbl_dir

    # 대상 디렉토리 준비
    dest_img_dirs = {}
    dest_lbl_dirs = {}
    for v in dest_versions:
        dimg = root / f"ODSR_v{v}"
        dlbl = root / f"ODSR_v{v}_anno"
        ensure_clean_dir(dimg, args.overwrite)
        ensure_clean_dir(dlbl, args.overwrite)
        dest_img_dirs[v] = dimg
        dest_lbl_dirs[v] = dlbl

    # 프레임 인덱스 구성
    frames = defaultdict(dict)
    print("[INFO] 원본 스캔 중...")
    for v in src_versions:
        files = list_images(src_img_dirs[v], exts)
        for img_path in files:
            base = img_path.stem  # v9_000003_jpg.rf.xxxxx
            frame_key = frame_key_from_filename(img_path.name)
            lbl_path = src_lbl_dirs[v] / f"{base}.txt"
            if not lbl_path.is_file():
                print(f"[WARN] 라벨 없음: {lbl_path}")
                continue
            frames[frame_key][v] = (img_path, lbl_path, img_path.name)

    # 공통 프레임만 사용
    eligible = [fk for fk, mp in frames.items() if len(mp) == K]
    if len(eligible) == 0:
        print("[ERROR] 사용할 프레임이 없습니다.", file=sys.stderr)
        sys.exit(1)
    eligible.sort()

    # CSV 로그
    csv_writer = None
    csv_file = None
    if args.csv_out:
        csv_file = open(args.csv_out, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_key", "dest_version", "src_version", "src_filename", "dest_filename"])

    print(f"[INFO] 재배치 시작 (프레임 수={len(eligible)}, 원본 버전 수={K}, 대상 시작=v{args.dest_start})")
    for fk in tqdm(eligible, desc="Assigning"):
        candidates = [(v, *frames[fk][v]) for v in src_versions]  # (src_v, img_path, lbl_path, fname)
        perm = list(range(K))
        random.shuffle(perm)
        for i, dest_v in enumerate(dest_versions):
            src_v, img_path, lbl_path, fname = candidates[perm[i]]

            # 대상 파일명 결정
            if args.keep_original_filenames:
                dest_stem = Path(fname).stem
            else:
                core = LEADING_VER_PREFIX_RE.sub("", Path(fname).stem)
                dest_stem = f"v{dest_v}_" + core

            dst_img = dest_img_dirs[dest_v] / f"{dest_stem}{img_path.suffix}"
            dst_lbl = dest_lbl_dirs[dest_v] / f"{dest_stem}.txt"

            copy_like(img_path, dst_img, args.link)
            copy_like(lbl_path, dst_lbl, args.link)

            if csv_writer:
                csv_writer.writerow([fk, dest_v, src_v, fname, dst_img.name])

    if csv_file:
        csv_file.close()

    # 검증 출력
    for v in dest_versions:
        n_img = sum(1 for _ in dest_img_dirs[v].glob("*"))
        n_lbl = sum(1 for _ in dest_lbl_dirs[v].glob("*.txt"))
        print(f"[OK] ODSR_v{v}: images={n_img}, annos={n_lbl}")
    print("[DONE] 재배치 완료.")

if __name__ == "__main__":
    main()
