# -*- coding: utf-8 -*-
"""YAML 생성 & YOLO 학습 스크립트 갱신 유틸"""
import os
import yaml
from pathlib import Path


# ─────────────────────────────────────────────
def create_yaml(version_tag: str, *, save_path: str, base_dir: str):
    """
    YOLO 데이터셋 YAML 파일 작성
    - dataset 경로: <base_dir>/ODSR-IHS_{version_tag}
    - 예: /home/jhcha2/jh_ws/yolo/datasets/ODSR-IHS_v9-v10_r60_m25
    """
    dataset_root = os.path.join(base_dir, f"ODSR-IHS_{version_tag}")
    fname = f"{version_tag}.yaml"
    path = os.path.join(save_path, fname)

    data = {
        # 절대 경로 ↓↓↓
        "train": os.path.join(dataset_root, "train"),
        "val":   os.path.join(dataset_root, "valid"),
        "test":  os.path.join(dataset_root, "test"),
        "nc": 14,
        "names": [
            "slippers", "stool", "wire", "carpet", "sofa", "socks", "feces",
            "table", "bed", "closetool", "book", "cabinet", "trashcan", "curtain",
        ],
        "roboflow": {           # 유지(필요 없으면 삭제해도 무방)
            "workspace": "agv",
            "project": "odsr-ihs",
            "version": 1,
            "license": "CC BY 4.0",
            "url": "https://universe.roboflow.com/agv/odsr-ihs/dataset/1",
        },
    }

    # YAML 저장
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    # names 한 줄로 강제
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            if line.startswith("names:"):
                f.write("names: ['slippers', 'stool', 'wire', 'carpet', 'sofa', "
                        "'socks', 'feces', 'table', 'bed', 'closetool', 'book', "
                        "'cabinet', 'trashcan', 'curtain']\n")
            elif line.strip().startswith("- "):
                continue
            else:
                f.write(line)

    print(f"✅ YAML 생성 완료: {path}")

def create_tirod_yaml(version_tag: str, *, save_path: str, base_dir: str):
    """
    TiROD용 YAML   (datasets/TiROD_<tag>/… 상대경로로 기록)
    """
    # 최종 저장 위치:  yolo_root/<tag>.yaml
    yaml_path = Path(save_path) / f"{version_tag}.yaml"

    # 데이터셋 폴더 이름 = TiROD_<tag>
    ds_folder = f"TiROD_{version_tag}"

    data = {
        "train": f"{ds_folder}/train",
        "val":   f"{ds_folder}/valid",
        "test":  f"{ds_folder}/test",
        "nc": 13,
        "names": [
            "bag", "bottle", "cardboard box", "chair", "potted plant",
            "traffic cone", "trashcan", "ball", "broom", "garden hose",
            "bucket", "bycicle", "gardening tool",
        ],
    }

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    print(f"✅ TiROD YAML 생성 완료: {yaml_path}")



# ─────────────────────────────────────────────
def update_data_yaml_in_script(script_path: str, yaml_filename: str):
    """
    YOLO 학습 스크립트(.py) 내부의 data="..." 라인을 yaml_filename 으로 치환
    """
    with open(script_path, "r") as f:
        lines = f.readlines()

    with open(script_path, "w") as f:
        for line in lines:
            if line.strip().startswith("data="):
                f.write(f'    data="{yaml_filename}",\n')
            else:
                f.write(line)
