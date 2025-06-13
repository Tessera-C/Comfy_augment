# -*- coding: utf-8 -*-
"""
범용 유틸
- TeeWithTimestamp : 터미널·로그 동시 출력
- get_auto_log_filename : 자동 로그 이름
- rename_latest_yolo_run : runs/detect/train* → train(<tag>)
- copy_results_csv       : results.csv 1 개 복사
- copy_run_outputs       : results.csv + best.pt + last.pt 복사
"""
from __future__ import annotations
import os, sys, datetime, shutil


# ─────────────────────────────────────────────
class TeeWithTimestamp:
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message: str):
        if message.strip():
            ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            message = "".join(ts + ln if ln.strip() else ln
                              for ln in message.splitlines(keepends=True))
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ─────────────────────────────────────────────
def get_auto_log_filename(version_tag: str, log_dir: str = "logs") -> str:
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"log_{version_tag}_{now}.txt")


# ─────────────────────────────────────────────
def rename_latest_yolo_run(work_dir: str, version_tag: str):
    """runs/detect/train* 중 최신 폴더를 train(<tag>)로 rename → 새 이름 리턴"""
    runs_detect = os.path.join(work_dir, "runs", "detect")
    if not os.path.isdir(runs_detect):
        print("[WARN] runs/detect 폴더 없음"); return None

    train_dirs = [d for d in os.listdir(runs_detect) if d.startswith("train")]
    if not train_dirs:
        print("[WARN] train* 폴더 없음"); return None

    latest_path = max((os.path.join(runs_detect, d) for d in train_dirs),
                      key=os.path.getmtime)
    new_name = f"train({version_tag})"
    new_path = os.path.join(runs_detect, new_name)
    idx = 1
    while os.path.exists(new_path):
        new_path = os.path.join(runs_detect, f"{new_name}_{idx}")
        idx += 1
    try:
        shutil.move(latest_path, new_path)
        print(f"[INFO] {os.path.basename(latest_path)} → {os.path.basename(new_path)}")
        return os.path.basename(new_path)
    except Exception as e:
        print(f"[WARN] rename 실패: {e}")
        return None


# ─────────────────────────────────────────────
def copy_results_csv(yolo_root: str, run_folder: str, logs_dir: str = "logs"):
    """results.csv 1 개만 logs/ 로 복사 (중복 시 건너뜀)"""
    src = os.path.join(yolo_root, "runs", "detect", run_folder, "results.csv")
    if not os.path.exists(src):
        print(f"[WARN] results.csv 없음: {src}"); return
    dst_dir = os.path.join(yolo_root, logs_dir)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, f"results_{run_folder}.csv")
    if os.path.exists(dst):
        print(f"[INFO] 이미 존재 → 생략: {dst}"); return
    shutil.copy2(src, dst)
    print(f"[INFO] results.csv 복사 완료 → {dst}")


# ─────────────────────────────────────────────
def copy_run_outputs(yolo_root: str, run_folder: str, results_dir: str = "results"):
    """
    runs/detect/<run_folder>/ 의
      - results.csv
      - weights/best.pt
    두 파일을 <results_dir>/ 로 복사 & 리네임.
    이미 존재하면 건너뜀.
    """
    base = os.path.join(yolo_root, "runs", "detect", run_folder)
    mapping = {
        os.path.join(base, "results.csv"):        f"results_{run_folder}.csv",
        os.path.join(base, "weights", "best.pt"): f"best_{run_folder}.pt",
    }
    dst_root = os.path.join(yolo_root, results_dir)
    os.makedirs(dst_root, exist_ok=True)

    for src, dst_name in mapping.items():
        if not os.path.exists(src):
            print(f"[WARN] 없음 → 생략: {src}")
            continue
        dst = os.path.join(dst_root, dst_name)
        if os.path.exists(dst):
            print(f"[INFO] 이미 존재 → 생략: {dst_name}")
            continue
        shutil.copy2(src, dst)
        print(f"[INFO] 복사 완료: {dst_name}")
