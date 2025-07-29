# data_pipeline/preprocess_tirod.py
from __future__ import annotations
import os, random, shutil, tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np

VALID_EXT = {".jpg", ".jpeg", ".png"}
DATASET_NAME = "TiROD"

# ──────────────────────────────────────────
def copy_and_prune_dataset_tirod(
    base_dir: str,
    ratio: float,
    *,
    seed: int = 42,
    version_tag: str,
    sampling: str = "random",           # "random" | "interval"
) -> None:
    """
    datasets/TiROD → datasets/TiROD_<version_tag> 복사 후
    train 하위 이미지를 ratio 만큼 삭제
    """
    src = Path(base_dir) / DATASET_NAME
    dst = Path(base_dir) / f"{DATASET_NAME}_{version_tag}"
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst)

    train_dir = dst / "train"
    imgs = sorted(f for f in os.listdir(train_dir) if Path(f).suffix.lower() in VALID_EXT)
    if not imgs:
        print(f"[WARN] '{train_dir}' 에 이미지가 없습니다."); return

    del_cnt = int(len(imgs) * ratio)
    if del_cnt == 0:
        print("[INFO] 삭제 비율이 낮아 0개 삭제되었습니다."); return

    # 삭제 대상 선택
    if sampling == "interval":
        idx = np.linspace(0, len(imgs) - 1, del_cnt, endpoint=False, dtype=int)
        targets = [imgs[i] for i in idx]
    else:
        random.seed(seed)
        targets = random.sample(imgs, del_cnt)

    # 실제 삭제
    for f in targets:
        (train_dir / f).unlink(missing_ok=True)
        (train_dir / Path(f).with_suffix(".txt")).unlink(missing_ok=True)

    print(f"'{src}' → '{dst}' 복사 및 {del_cnt}개 파일 삭제 완료 (sampling={sampling})")

# ──────────────────────────────────────────
def copy_augmented_files_tirod(
    train_dir: str,
    versions: List[str] | str,
    *,
    match_ratio: float = 1.0,
    base_output: str = "output",
    seed: int = 42,
):
    """
    라벨이 없으면 0-byte .txt 를 만들어 주는 TiROD 전용 증강본 매칭 복사
    """
    if isinstance(versions, str): versions = [versions]
    train_dir_p = Path(train_dir)
    random.seed(seed)

    # 접두어 없는 원본 베이스
    base_names = [
        f.stem for f in train_dir_p.iterdir()
        if f.suffix.lower() in VALID_EXT and not any(f.name.startswith(f"{v}_") for v in versions)
    ]

    candidates: List[Tuple[str,str]] = []   # (base, version)
    for v in versions:
        img_dir = Path(base_output) / f"{DATASET_NAME}_{v}"
        pref = f"{v}_"
        for n in base_names:
            if any((img_dir / f"{pref}{n}{ext}").exists() for ext in VALID_EXT):
                candidates.append((n, v))

    target = min(int(len(base_names) * match_ratio), len(candidates))
    picked  = random.sample(candidates, target)

    def _copy(src: Path):
        shutil.copy(src, train_dir_p / src.name)

    for n, v in picked:
        img_dir = Path(base_output) / f"{DATASET_NAME}_{v}"
        lbl_dir = Path(base_output) / f"{DATASET_NAME}_{v}_anno"
        pref = f"{v}_"

        # 이미지
        for ext in VALID_EXT:
            fp = img_dir / f"{pref}{n}{ext}"
            if fp.exists(): _copy(fp); break
        # 라벨
        lbl_src = lbl_dir / f"{pref}{n}.txt"
        lbl_dst = train_dir_p / f"{pref}{n}.txt"
        if lbl_src.exists():
            _copy(lbl_src)
        else:
            lbl_dst.touch()

    actual = target / max(1, len(base_names))
    print(f"[INFO] 접두어 복사 {target}쌍 / {len(base_names)} (actual match-ratio≈{actual:.2f})")

