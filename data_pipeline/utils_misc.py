# -*- coding: utf-8 -*-
"""
범용 유틸 모음
- TeeWithTimestamp : 터미널‧로그 파일 동시 출력 + 타임스탬프
- get_auto_log_filename : 날짜가 포함된 자동 로그 이름
- rename_latest_yolo_run : runs/detect/train* → train(<version_tag>) 로 변경
"""
from __future__ import annotations
import os
import sys
import datetime
import shutil


# ─────────────────────────────────────────────
class TeeWithTimestamp:
    """터미널 + 로그 파일 동시 출력, 줄마다 타임스탬프 부여"""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message: str):
        if message.strip():  # 빈 줄(공백·개행) 제외
            ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            message = "".join(
                ts + ln if ln.strip() else ln for ln in message.splitlines(keepends=True)
            )
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ─────────────────────────────────────────────
def get_auto_log_filename(version_tag: str, log_dir: str = "logs") -> str:
    """logs/log_<version_tag>_<timestamp>.txt 경로 반환"""
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"log_{version_tag}_{now}.txt")


# ─────────────────────────────────────────────
def rename_latest_yolo_run(work_dir: str, version_tag: str):
    """
    work_dir/runs/detect 에서 가장 최근 'train*' 폴더를
    'train(<version_tag>)' 로 변경
    """
    runs_detect = os.path.join(work_dir, "runs", "detect")
    if not os.path.isdir(runs_detect):
        print("[WARN] runs/detect 폴더를 찾지 못했습니다.")
        return

    train_dirs = [d for d in os.listdir(runs_detect) if d.startswith("train")]
    if not train_dirs:
        print("[WARN] train* 폴더가 존재하지 않습니다.")
        return

    latest_path = max(
        (os.path.join(runs_detect, d) for d in train_dirs),
        key=os.path.getmtime,
    )
    new_name = f"train({version_tag})"
    new_path = os.path.join(runs_detect, new_name)

    # 동일 이름 존재 시 뒤에 _1, _2… 증분
    idx = 1
    while os.path.exists(new_path):
        new_path = os.path.join(runs_detect, f"{new_name}_{idx}")
        idx += 1

    try:
        shutil.move(latest_path, new_path)
        print(
            f"[INFO] runs/detect/{os.path.basename(latest_path)} → {os.path.basename(new_path)}"
        )
    except Exception as e:
        print(f"[WARN] 폴더 이름 변경 실패: {e}")
