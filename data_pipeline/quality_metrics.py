# -*- coding: utf-8 -*-
"""
품질 지표 계산 & LPIPS-기반 필터링 유틸
  · FID (데이터셋 단위)
  · LPIPS (이미지 단위)
  · LPIPS-구간 필터  : range / top / bottom / split-k
"""
from __future__ import annotations

import os, subprocess, tempfile, shutil, csv, pathlib, json
from pathlib import Path
from typing import List, Tuple

import torch, lpips, torchvision.transforms as TF
import numpy as np
import pandas as pd
from PIL import Image

# ───────────────────────────── 공통 ──────────────────────────────
VALID_EXT = {".jpg", ".jpeg", ".png"}


# ───────────────────────────── FID ──────────────────────────────
def compute_fid(dir0: str, dir1: str) -> float:
    """pytorch-fid 모듈 호출 → FID float 반환"""
    cmd = [
        "python",
        "-m",
        "pytorch_fid",
        dir0,
        dir1,
        "--device",
        "cuda" if torch.cuda.is_available() else "cpu",
    ]
    out = subprocess.check_output(cmd, text=True)
    for line in out.splitlines():
        if line.startswith("FID:"):
            return float(line.split()[-1])
    raise RuntimeError("FID 파싱 실패")


# ────────────────── LPIPS : 모든 짝 계산 & 통계 ──────────────────
def compute_lpips_per_image(
    dir_orig: str,
    dir_gen: str,
    *,
    prefix: str = "",
    batch_size: int = 16,
    net: str = "vgg",
) -> Tuple[list[Tuple[str, float]], dict]:
    """
    · dir_orig 에 있는 원본 이미지와 dir_gen 의 {prefix}{name}.ext 를 1:1 비교
    · 반환
        - pairs : [(base_name, lpips_score), ...]
        - stats : {mean, median, std, min, max, n_pairs}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net=net).to(device).eval()

    filenames = [
        f for f in os.listdir(dir_orig) if Path(f).suffix.lower() in VALID_EXT
    ]

    pairs, scores = [], []
    b0, b1, names = [], [], []

    def _flush():
        if not b0:
            return
        with torch.no_grad():
            s = (
                loss_fn(torch.cat(b0).to(device), torch.cat(b1).to(device))
                .squeeze()
                .cpu()
                .numpy()
            )
        for n, sc in zip(names, np.atleast_1d(s)):
            pairs.append((n, float(sc)))
            scores.append(float(sc))
        b0.clear(), b1.clear(), names.clear()
        torch.cuda.empty_cache()

    for fn in filenames:
        base, _ = os.path.splitext(fn)
        src = os.path.join(dir_orig, fn)
        tgt = next(
            (
                os.path.join(dir_gen, prefix + base + ext)
                for ext in VALID_EXT
                if os.path.exists(os.path.join(dir_gen, prefix + base + ext))
            ),
            None,
        )
        if tgt is None:
            continue

        b0.append(TF.ToTensor()(Image.open(src).convert("RGB")).unsqueeze(0))
        b1.append(TF.ToTensor()(Image.open(tgt).convert("RGB")).unsqueeze(0))
        names.append(base)

        if len(b0) == batch_size:
            _flush()
    _flush()

    if not scores:
        raise RuntimeError("LPIPS 비교 쌍이 없습니다.")

    arr = np.array(scores)
    stats = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_pairs": int(arr.size),
    }
    return pairs, stats


# ───────────── train 폴더 → orig / gen  임시 분리 ────────────────
def _split_dataset_for_fid(train_dir: str, prefix: str):
    tmp_root = tempfile.mkdtemp()
    dir_o, dir_g = Path(tmp_root, "orig"), Path(tmp_root, "gen")
    dir_o.mkdir(); dir_g.mkdir()

    for f in os.listdir(train_dir):
        ext = Path(f).suffix.lower()
        if ext not in VALID_EXT or f.startswith(prefix):
            continue
        base, _ = os.path.splitext(f)
        gen = next(
            (
                Path(train_dir, f"{prefix}{base}{e}")
                for e in VALID_EXT
                if Path(train_dir, f"{prefix}{base}{e}").exists()
            ),
            None,
        )
        if gen:
            shutil.copy(Path(train_dir, f), dir_o / f)
            shutil.copy(gen, dir_g / f)

    return str(dir_o), str(dir_g), tmp_root


def compute_fid_per_versions(
    train_dir: str, versions: List[str]
) -> Tuple[dict, dict]:
    """
    버전별(prefix별) FID  {ver: score},  매칭 없음 {ver: (n_orig, n_gen)}
    """
    results, skipped = {}, {}
    for v in versions:
        prefix = f"{v}_"
        dir_o, dir_g, tmp = _split_dataset_for_fid(train_dir, prefix)
        n_o, n_g = len(os.listdir(dir_o)), len(os.listdir(dir_g))
        if not n_o or not n_g:
            skipped[v] = (n_o, n_g)
        else:
            results[v] = compute_fid(dir_o, dir_g)
        shutil.rmtree(tmp)
    return results, skipped


# ─────────────────────── LPIPS 기반 데이터셋 필터 ───────────────────────
def filter_dataset_by_lpips(
    *,
    csv_path: str,
    train_dir: str,
    prefix: str,
    mode: str = "range",  # "range" | "top" | "bottom" | "split"
    lpips_min: float | None = None,
    lpips_max: float | None = None,
    percent: float | None = None,
    split_k: int | None = None,
    split_idx: int | None = None,
    tag_suffix: str = "lpipsflt",
) -> tuple[str, int, int]:
    """
    LPIPS 기준으로 접두어 이미지를 골라 **새 데이터셋**(tmp_dir)을 만든다.
      · return  (tmp_root_path, n_pref, n_orig)
    """
    df = pd.read_csv(csv_path, names=["filename", "lpips"], header=0)

    # 1) 접두어 base 이름 필터링 --------------------------------------------------
    if mode == "range":
        if lpips_min is None or lpips_max is None:
            raise ValueError("--lpips-min / --lpips-max 모두 필요")
        sel = df[(df.lpips >= lpips_min) & (df.lpips <= lpips_max)]

    elif mode in ("top", "bottom"):
        if percent is None:
            raise ValueError("--lpips-percent 필요")
        k = max(1, int(len(df) * percent / 100))
        sel = df.sort_values("lpips", ascending=(mode == "bottom")).head(k)

    elif mode == "split":
        if split_k is None or split_idx is None:
            raise ValueError("--lpips-split & --lpips-split-idx 필요")
        if not (0 <= split_idx < split_k):
            raise ValueError("split-idx 범위 오류")
        df_sorted = df.sort_values("lpips", ascending=True, ignore_index=True)
        group = len(df_sorted) // split_k
        start = split_idx * group
        end = start + group if split_idx < split_k - 1 else len(df_sorted)
        sel = df_sorted.iloc[start:end]

    else:
        raise ValueError(f"mode '{mode}' 지원 안 함")

    if sel.empty:
        raise RuntimeError("선택된 이미지가 없습니다.")

    bases_pref = set(sel.filename)

    # 2) 새 폴더 ---------------------------------------------------------------
    tmp_root = tempfile.mkdtemp(prefix=f"{tag_suffix}_")
    new_train = Path(tmp_root, "train"); new_train.mkdir()

    def _copy_pair(img_path: Path):
        dst_img = new_train / img_path.name
        shutil.copy(img_path, dst_img)
        lbl = img_path.with_suffix(".txt")
        if lbl.exists():
            shutil.copy(lbl, new_train / lbl.name)

    n_pref = n_orig = 0

    # (a) 원본 이미지 전부 복사
    for f in Path(train_dir).iterdir():
        if f.suffix.lower() not in VALID_EXT or f.name.startswith(prefix):
            continue
        _copy_pair(f)
        n_orig += 1

    # (b) 선택된 접두어 이미지 복사
    for base in bases_pref:
        for ext in VALID_EXT:
            cand = Path(train_dir, f"{prefix}{base}{ext}")
            if cand.exists():
                _copy_pair(cand); n_pref += 1; break

    # valid / test 는 심볼릭 링크
    parent = Path(train_dir).parent
    for split in ("valid", "test"):
        src = parent / split
        if src.exists():
            (Path(tmp_root) / split).symlink_to(src)

    return tmp_root, n_pref, n_orig
