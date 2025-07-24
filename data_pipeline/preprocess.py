# -*- coding: utf-8 -*-
"""
전처리 유틸 집합
- add_prefix_to_filenames
- copy_and_prune_dataset
- copy_augmented_files : 다중 버전 & match_ratio≥1.0 섞기 지원
"""
import os
import random
import shutil
import numpy as np
from typing import List
from pathlib import Path

VALID_EXT = {".jpg", ".jpeg", ".png"}

# ─────────────────────────────────────────────
def add_prefix_to_filenames(version: str, source_folder: str, destination_folder: str):
    """
    `source_folder` 전체를 `destination_folder` 로 복사한 뒤
    모든 파일 이름 앞에 `{version}_` 접두어를 붙인다.
    """
    import jh_filename  # 사용자 정의 유틸

    if not os.path.isdir(source_folder):
        print(f"[WARN] '{source_folder}' 경로가 존재하지 않습니다.")
        return

    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(source_folder, destination_folder)
    jh_filename.prepend_text_to_filenames(destination_folder, f"{version}_")
    print(f"'{source_folder}' → '{destination_folder}' 복사 및 접두어 추가 완료")


# ─────────────────────────────────────────────
def copy_and_prune_dataset(
    base_dir: str, version: str, ratio: float, seed: int = 42, *, version_tag: str, dataset_name: str = "ODSR-IHS", sampling: str = "random"
):
    """
    datasets/ODSR-IHS  →  datasets/ODSR-IHS_{version_tag} 로 복사 후
    train 하위 이미지·라벨을 ratio만큼 무작위 삭제
    """
    src = os.path.join(base_dir, dataset_name)
    dst = os.path.join(base_dir, f"{dataset_name}_{version_tag}")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    train_dir = os.path.join(dst, "train")
    imgs = sorted(
        [f for f in os.listdir(train_dir) if f.lower().endswith(tuple(VALID_EXT))]
    )
    del_cnt = int(len(imgs) * ratio)

    if sampling == "interval":
        # 균등 분포 index 계산 → 삭제 대상
        idx = np.linspace(0, len(imgs) - 1, del_cnt, dtype=int, endpoint=False)
        targets = [imgs[i] for i in idx]
    else:  # random
        random.seed(seed)
        targets = random.sample(imgs, del_cnt)

    for f in targets:
        os.remove(os.path.join(train_dir, f))
        lbl = os.path.join(train_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl):
            os.remove(lbl)

    print(
        f"'{src}' → '{dst}' 복사 및 {del_cnt}개 파일 삭제 완료 "
        f"(sampling={sampling})"
    )


# ─────────────────────────────────────────────
def _copy_pairs(names, v, img_src, lbl_src, train_dir, append):
    """이미지·라벨 쌍 복사 내부 함수"""
    for n in names:
        # 이미지
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(img_src, append + n + ext)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(train_dir, append + n + ext))
                break
        # 라벨
        shutil.copy2(
            os.path.join(lbl_src, append + n + ".txt"),
            os.path.join(train_dir, append + n + ".txt"),
        )

def _copy_one(src: str, dst_dir: str):
    """파일을 dst_dir 로 복사(동일 이름 존재 시 덮어쓰기)."""
    shutil.copy(src, os.path.join(dst_dir, os.path.basename(src)))


def copy_augmented_files(
    train_dir: str,
    versions: List[str] | str,
    *,
    match_ratio: float = 1.0,
    base_output: str = "output",
    seed: int = 42,
):
    """
    train_dir 의 '원본(접두어 없는)' 이미지를 기준으로
    - 단일 버전 또는 match_ratio<1.0 ➜ 기존 방식
    - 다중 버전 & match_ratio≥1.0 ➜ 버전을 섞어 무작위 추출

    복사 시 이미지·라벨 쌍을 함께 가져온다.
    """
    if isinstance(versions, str):
        versions = [versions]

    random.seed(seed)

    # ── (0) 원본 파일명 목록 ─────────────────────────────
    base_names = [
        f.stem
        for f in Path(train_dir).iterdir()
        if f.suffix.lower() in VALID_EXT
        and not any(f.name.startswith(f"{v}_") for v in versions)
    ]

    # 단일 버전 이거나 match_ratio<1.0 ------------------
    if len(versions) == 1 or match_ratio < 1.0:
        v = versions[0]
        img_src_dir = Path(base_output) / f"ODSR_{v}"
        lbl_src_dir = Path(base_output) / f"ODSR_{v}_anno"
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
                    _copy_one(fp, train_dir)
                    break
            _copy_one(lbl_src_dir / f"{pref}{n}.txt", train_dir)

        print(f"[INFO] 접두어 복사 {k}쌍 ({v}, match_ratio={match_ratio})")
        return

    # 다중 버전 & match_ratio ≥1.0 -----------------------
    candidates: list[tuple[str, str]] = []         # (base, version)
    for v in versions:
        img_dir = Path(base_output) / f"ODSR_{v}"
        lbl_dir = Path(base_output) / f"ODSR_{v}_anno"
        pref = f"{v}_"

        for n in base_names:
            if any((img_dir / f"{pref}{n}{ext}").exists() for ext in VALID_EXT) \
               and (lbl_dir / f"{pref}{n}.txt").exists():
                candidates.append((n, v))

    if not candidates:
        print("[WARN] 매칭 가능한 접두어 파일이 없습니다."); return

    target = min(int(len(base_names) * match_ratio), len(candidates))
    picked  = random.sample(candidates, target)

    for n, v in picked:
        img_dir = Path(base_output) / f"ODSR_{v}"
        lbl_dir = Path(base_output) / f"ODSR_{v}_anno"
        pref = f"{v}_"

        for ext in VALID_EXT:
            fp = img_dir / f"{pref}{n}{ext}"
            if fp.exists():
                _copy_one(fp, train_dir); break
        _copy_one(lbl_dir / f"{pref}{n}.txt", train_dir)

    actual = target / max(1, len(base_names))
    sat = " (saturated)" if target < int(len(base_names) * match_ratio) else ""
    print(f"[INFO] 접두어 복사 {target}쌍 / {len(base_names)} "
          f"(actual match-ratio≈{actual:.2f}){sat}")
