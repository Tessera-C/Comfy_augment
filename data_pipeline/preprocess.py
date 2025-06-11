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
from typing import List


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
    base_dir: str, version: str, ratio: float, seed: int = 42, *, version_tag: str
):
    """
    datasets/ODSR-IHS  →  datasets/ODSR-IHS_{version_tag} 로 복사 후
    train 하위 이미지·라벨을 ratio만큼 무작위 삭제
    """
    src = os.path.join(base_dir, "ODSR-IHS")
    dst = os.path.join(base_dir, f"ODSR-IHS_{version_tag}")

    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    train_dir = os.path.join(dst, "train")
    imgs = [f for f in os.listdir(train_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.seed(seed)
    del_cnt = int(len(imgs) * ratio)

    for f in random.sample(imgs, del_cnt):
        os.remove(os.path.join(train_dir, f))
        txt = os.path.join(train_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(txt):
            os.remove(txt)

    print(f"'{src}' → '{dst}' 복사 및 {del_cnt}개 파일 삭제 완료")


# ─────────────────────────────────────────────
def copy_augmented_files(
    train_dir: str,
    versions: List[str] | str,
    *,
    match_ratio: float = 1.0,
    base_output: str = "output",
    seed: int = 42,
):
    """
    train_dir (접두어 없는 원본 이미지) 기준으로 접두어 파일 복사
    - versions : ['v9', 'v10', ...] 또는 'v9'
    - match_ratio <1.0 또는 len(versions)==1 → 단일 버전 기존 방식
    - match_ratio ≥1.0 & len(versions)>1   → 서로 다른 버전 섞어 복사
    """
    if isinstance(versions, str):
        versions = [versions]

    random.seed(seed)
    
    # 접두어 없는 파일 이름 리스트
    base_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(train_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and all(not f.startswith(f"{v}_") for v in versions)
    ]

    # ────────── 1) 단일 버전 또는 match_ratio<1.0 ──────────
    if len(versions) == 1 or match_ratio < 1.0:
        v = versions[0]
        img_src = os.path.join(base_output, f"ODSR_{v}")
        lbl_src = os.path.join(base_output, f"ODSR_{v}_anno")
        append = f"{v}_"

        valid = [
            n
            for n in base_names
            if any(os.path.exists(os.path.join(img_src, append + n + ext)) for ext in [".jpg", ".jpeg", ".png"])
            and os.path.exists(os.path.join(lbl_src, append + n + ".txt"))
        ]

        k = int(len(valid) * match_ratio)
        selected = random.sample(valid, k)

        _copy_pairs(selected, v, img_src, lbl_src, train_dir, append)
        print(f"복사 완료: {len(selected)} 쌍 ({v})")
        return

    # ────────── 2) 다중 버전 & match_ratio ≥1.0 ──────────
    candidate = []  # (name, version)
    for v in versions:
        img_src = os.path.join(base_output, f"ODSR_{v}")
        lbl_src = os.path.join(base_output, f"ODSR_{v}_anno")
        append = f"{v}_"
        for n in base_names:
            img_ok = any(os.path.exists(os.path.join(img_src, append + n + ext)) for ext in [".jpg", ".jpeg", ".png"])
            lbl_ok = os.path.exists(os.path.join(lbl_src, append + n + ".txt"))
            if img_ok and lbl_ok:
                candidate.append((n, v))

    if not candidate:
        print("[WARN] 일치하는 접두어 파일을 찾지 못했습니다.")
        return

    target_cnt = int(len(base_names) * match_ratio)
    target_cnt = min(target_cnt, len(candidate))
    picked = random.sample(candidate, target_cnt)

    copied = 0
    for n, v in picked:
        img_src = os.path.join(base_output, f"ODSR_{v}")
        lbl_src = os.path.join(base_output, f"ODSR_{v}_anno")
        append = f"{v}_"
        _copy_pairs([n], v, img_src, lbl_src, train_dir, append)
        copied += 1

    print(f"복사 완료: {copied} 쌍 (versions={','.join(versions)})")


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
