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
from dreamsim import dreamsim
from torchvision import transforms

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

def compute_metric_per_image(
    dir_orig: str,
    dir_gen: str,
    *,
    prefix: str = "",
    batch_size: int = 16,
    metric: str = "lpips",   # "lpips" | "dreamsim"
    lpips_net: str = "vgg",  # LPIPS용
) -> Tuple[list[Tuple[str, float]], dict]:
    """
    metric 선택에 따라 이미지-유사도(LPIPS 또는 DreamSim) 1 : 1 계산.
      · return
          - pairs : [(base_name, score), ...]
          - stats : {mean, median, std, min, max, n_pairs}
    """
    metric = metric.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── ① 모델·전처리 준비 ──────────────────────────────────────────
    if metric == "lpips":
        loss_fn = lpips.LPIPS(net=lpips_net).to(device).eval()
        transform = transforms.ToTensor()                 # 그대로 사용
    elif metric == "dreamsim":
        loss_fn, _ = dreamsim(pretrained=True, device=device)  # distance 함수
        img_size = 224
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError("metric은 'lpips' 또는 'dreamsim'만 지원합니다.")

    # ── ② 파일 목록 수집 ───────────────────────────────────────────
    filenames = [
        f for f in os.listdir(dir_orig) if Path(f).suffix.lower() in VALID_EXT
    ]

    pairs, scores = [], []
    b0, b1, names = [], [], []

    # ── ③ 배치 단위 계산 함수 ──────────────────────────────────────
    def _flush():
        if not b0:
            return
        with torch.no_grad():
            if metric == "lpips":
                s = (
                    loss_fn(torch.cat(b0).to(device),
                            torch.cat(b1).to(device))
                    .squeeze()
                    .cpu()
                    .numpy()
                )
            else:  # dreamsim
                s = []
                for x, y in zip(b0, b1):
                    s.append(float(loss_fn(x.to(device), y.to(device)).item()))
                s = np.array(s)

        for n, sc in zip(names, np.atleast_1d(s)):
            pairs.append((n, float(sc)))
            scores.append(float(sc))
        b0.clear(), b1.clear(), names.clear()
        torch.cuda.empty_cache()

    # ── ④ 이미지 쌍 만들기 ────────────────────────────────────────
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

        img0 = transform(Image.open(src).convert("RGB")).unsqueeze(0)
        img1 = transform(Image.open(tgt).convert("RGB")).unsqueeze(0)
        b0.append(img0)
        b1.append(img1)
        names.append(base)

        if len(b0) == batch_size:
            _flush()
    _flush()

    if not scores:
        raise RuntimeError(f"{metric.upper()} 비교 쌍이 없습니다.")

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
def filter_dataset_by_metric(
    *,
    csv_path: str,
    train_dir: str,
    prefixes: list[str],          # v9, v10, ...
    metric_name: str,
    mode: str = "range",
    lpips_min: float | None = None,
    lpips_max: float | None = None,
    percent: float | None = None,
    split_k: int | None = None,
    split_idx: int | None = None,
    tag_suffix: str = "flt",
) -> tuple[str, int, int]:
    """모든 증강본 접두어를 대상으로 점수 필터링."""
    # --- CSV 로드 & 'score' 열 통일 -----------------------------------
    df = pd.read_csv(csv_path)
    if metric_name not in df.columns:
        raise ValueError(f"CSV에 '{metric_name}' 열이 없습니다.")
    df = df.rename(columns={metric_name: "score"})

    # --- 선택 행 ------------------------------------------------------
    if mode == "range":
        if lpips_min is None or lpips_max is None:
            raise ValueError("--lpips-min / --lpips-max 모두 필요")
        sel = df[(df.score >= lpips_min) & (df.score <= lpips_max)]

    elif mode in ("top", "bottom"):
        if percent is None:
            raise ValueError("--lpips-percent 필요")
        k = max(1, int(len(df) * percent / 100))
        sel = df.sort_values("score", ascending=(mode == "bottom")).head(k)

    elif mode == "split":
        if split_k is None or split_idx is None:
            raise ValueError("--lpips-split & --lpips-split-idx 필요")
        if not (0 <= split_idx < split_k):
            raise ValueError("split-idx 범위 오류")
        df_sorted = df.sort_values("score", ignore_index=True)
        group = len(df_sorted) // split_k
        start = split_idx * group
        end = start + group if split_idx < split_k - 1 else len(df_sorted)
        sel = df_sorted.iloc[start:end]

    else:
        raise ValueError(f"mode '{mode}' 지원 안 함")
    if sel.empty:
        raise RuntimeError("선택된 이미지가 없습니다.")

    sel_names = set(sel.filename)          # 'v9_xxx' 형식 포함

    # ── 선택된 점수 통계 --------------------------------------------
    sel_scores = sel.score.to_numpy()

    sel_stats = {
    "mean":   float(sel_scores.mean()),
    "median": float(np.median(sel_scores)),
    "std":    float(sel_scores.std()),
    "min":    float(sel_scores.min()),
    "max":    float(sel_scores.max()),
    "n_pairs": int(sel_scores.size),
    }

    # --- 새 데이터셋 폴더 --------------------------------------------
    tmp_root = tempfile.mkdtemp(prefix=f"{tag_suffix}_")
    new_train = Path(tmp_root, "train"); new_train.mkdir()

    def _copy_pair(img_path: Path):
        dst = new_train / img_path.name
        shutil.copy(img_path, dst)
        lbl = img_path.with_suffix(".txt")
        if lbl.exists():
            shutil.copy(lbl, new_train / lbl.name)

    # (a) 원본 이미지 복사(접두어 없는)
    for f in Path(train_dir).iterdir():
        if f.suffix.lower() not in VALID_EXT:
            continue
        has_pref = any(f.name.startswith(f"{p}_") for p in prefixes)
        if not has_pref:                   # 원본
            _copy_pair(f)

    # (b) 필터 통과한 증강본 복사
    for fname in sel_names:
        for ext in VALID_EXT:
            cand = Path(train_dir, f"{fname}{ext}")
            if cand.exists():
                _copy_pair(cand); break

    # valid/test 링크 유지
    parent = Path(train_dir).parent
    for split in ("valid", "test"):
        src = parent / split
        if src.exists():
            (Path(tmp_root) / split).symlink_to(src)

    # 통계
    n_pref = len(sel_names)
    n_orig = sum(1 for f in new_train.iterdir()
                 if f.suffix.lower() in VALID_EXT and
                 not any(f.name.startswith(f"{p}_") for p in prefixes))
    return tmp_root, n_pref, n_orig, sel_stats