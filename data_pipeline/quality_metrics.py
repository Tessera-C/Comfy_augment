# -*- coding: utf-8 -*-
import os, subprocess, json, torch, lpips, tempfile, shutil, csv, pathlib
import torchvision.transforms as TF
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

VALID_EXT = {".jpg", ".jpeg", ".png"}

# ───────────────────────────────────────── FID
def compute_fid(dir0: str, dir1: str) -> float:
    cmd = ["python", "-m", "pytorch_fid", dir0, dir1,
           "--device", "cuda" if torch.cuda.is_available() else "cpu"]
    out = subprocess.check_output(cmd, text=True)
    for line in out.splitlines():
        if line.startswith("FID:"):
            return float(line.split()[-1])
    raise RuntimeError("FID 파싱 실패")

# ───────────────────────────── LPIPS 한 이미지씩
def compute_lpips_per_image(
    dir0: str, dir1: str, prefix: str = "", batch_size: int = 16
) -> tuple[list[tuple[str, float]], dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = lpips.LPIPS(net="vgg").to(device)

    filenames = [f for f in os.listdir(dir0) if Path(f).suffix.lower() in VALID_EXT]
    pairs, scores, b0, b1, names = [], [], [], [], []

    def _flush():
        if not b0:
            return
        with torch.no_grad():
            s = loss_fn(torch.cat(b0).to(device),
                        torch.cat(b1).to(device)).squeeze().cpu().numpy()
        for n, sc in zip(names, np.atleast_1d(s)):
            pairs.append((n, float(sc))); scores.append(float(sc))
        b0.clear(); b1.clear(); names.clear()

    for fn in filenames:
        name, _ = os.path.splitext(fn)
        src = os.path.join(dir0, fn)
        tgt = next((os.path.join(dir1, prefix + name + e)
                    for e in VALID_EXT if os.path.exists(os.path.join(dir1, prefix + name + e))), None)
        if tgt is None:
            continue
        b0.append(TF.ToTensor()(Image.open(src).convert("RGB")).unsqueeze(0))
        b1.append(TF.ToTensor()(Image.open(tgt).convert("RGB")).unsqueeze(0))
        names.append(name)
        if len(b0) == batch_size:
            _flush()
    _flush()
    if not scores:
        raise RuntimeError("LPIPS 비교 쌍이 없습니다.")

    arr = np.array(scores)
    stats = dict(mean=float(arr.mean()), median=float(np.median(arr)),
                 std=float(arr.std()),  min=float(arr.min()),
                 max=float(arr.max()),  n_pairs=len(arr))
    return pairs, stats

# ───────────────────────────── train 폴더 → orig/gen 임시분리
def split_dataset_for_fid(train_dir: str, prefix: str):
    tmp_root = tempfile.mkdtemp()
    dir_o = os.path.join(tmp_root, "orig"); os.makedirs(dir_o)
    dir_g = os.path.join(tmp_root, "gen");  os.makedirs(dir_g)

    for f in os.listdir(train_dir):
        ext = Path(f).suffix.lower()
        if ext not in VALID_EXT or f.startswith(prefix):
            continue
        base, _ = os.path.splitext(f)
        gen = next((os.path.join(train_dir, prefix + base + e)
                    for e in VALID_EXT if os.path.exists(os.path.join(train_dir, prefix + base + e))), None)
        if gen:
            shutil.copy(os.path.join(train_dir, f),  os.path.join(dir_o, f))
            shutil.copy(gen,                         os.path.join(dir_g, f))
    return dir_o, dir_g, tmp_root

# ───────────────────────────── 버전별 FID
def compute_fid_per_versions(train_dir: str, versions: list[str]):
    res, skip = {}, {}
    for v in versions:
        prefix = f"{v}_"
        o,g,tmp = split_dataset_for_fid(train_dir, prefix)
        if not os.listdir(o) or not os.listdir(g):
            skip[v] = (len(os.listdir(o)), len(os.listdir(g))); shutil.rmtree(tmp); continue
        res[v] = compute_fid(o, g); shutil.rmtree(tmp)
    return res, skip

# ───────────────────────────── LPIPS 필터
def filter_dataset_by_lpips(
    csv_path: str,
    train_dir: str,
    prefix: str,
    mode: str = "range",
    lpips_min: float | None = None,
    lpips_max: float | None = None,
    percent: float | None = None,
) -> tuple[str, int, int]:
    """
    새 데이터셋(tmp_root) 경로와
      n_pref : 필터링된 접두어 이미지 수
      n_orig : 복사된 원본 이미지 수
    를 반환
    """
    df = pd.read_csv(csv_path)        # cols: filename, lpips
    # ── 1) LPIPS 조건으로 접두어 base 이름 선택 ────────────────
    if mode == "range":
        sel = df[(df.lpips >= lpips_min) & (df.lpips <= lpips_max)]
    else:
        k = max(1, int(len(df) * percent / 100))
        sel = df.sort_values("lpips", ascending=(mode == "bottom")).head(k)
    bases_pref = set(sel.filename)

    # ── 2) 새 폴더 생성 ───────────────────────────────────────
    tmp_root = tempfile.mkdtemp(prefix="lpipsflt_")
    new_train = os.path.join(tmp_root, "train"); os.makedirs(new_train)

    def _copy(img_path: str):
        if not os.path.exists(img_path):
            return
        shutil.copy(img_path, os.path.join(new_train, os.path.basename(img_path)))
        lbl = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(lbl):
            shutil.copy(lbl, os.path.join(new_train, os.path.basename(lbl)))

    n_pref = n_orig = 0

    # ── 3) (a) 원본: 접두어 없는 모든 이미지+라벨 복사 ───────
    for f in os.listdir(train_dir):
        if f.startswith(prefix):
            continue
        if Path(f).suffix.lower() in VALID_EXT:
            _copy(os.path.join(train_dir, f))
            n_orig += 1
        elif f.endswith(".txt"):      # 라벨은 이미지 복사할 때 함께 처리되므로 skip
            continue

    #      (b) 선택된 접두어 이미지+라벨 복사 ───────────────
    for base in bases_pref:
        for ext in VALID_EXT:
            _copy(os.path.join(train_dir, prefix + base + ext))
            if os.path.exists(os.path.join(train_dir, prefix + base + ext)):
                n_pref += 1
                break   # 찾으면 ext 루프 탈출

    # ── 4) valid/test 링크 ───────────────────────────────────
    parent = Path(train_dir).parent
    for split in ("valid", "test"):
        src = parent / split
        if src.exists():
            (Path(tmp_root) / split).symlink_to(src)

    return tmp_root, n_pref, n_orig

