# -*- coding: utf-8 -*-
"""YAML 생성 & YOLO 학습 스크립트 갱신 유틸"""
import os
import yaml


# ─────────────────────────────────────────────
def create_yaml(version_tag: str, delete_ratio: float, match_ratio: float, *, save_path: str):
    """
    datasets/ODSR-IHS_{version_tag} 경로를 가리키는 YOLO 데이터 YAML 파일 작성
    version_tag 예: v9-v10_r50_m150
    """
    fname = f"{version_tag}.yaml"
    path = os.path.join(save_path, fname)
    dataset = f"ODSR-IHS_{version_tag}"

    data = {
        "train": f"{dataset}/train",
        "val": f"{dataset}/valid",
        "test": f"{dataset}/test",
        "nc": 14,
        "names": [
            "slippers",
            "stool",
            "wire",
            "carpet",
            "sofa",
            "socks",
            "feces",
            "table",
            "bed",
            "closetool",
            "book",
            "cabinet",
            "trashcan",
            "curtain",
        ],
        "roboflow": {
            "workspace": "agv",
            "project": "odsr-ihs",
            "version": 1,
            "license": "CC BY 4.0",
            "url": "https://universe.roboflow.com/agv/odsr-ihs/dataset/1",
        },
    }

    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    # names 한 줄로 고정 출력
    with open(path, "r") as f:
        lines = f.readlines()
    with open(path, "w") as f:
        for line in lines:
            if line.startswith("names:"):
                f.write(
                    "names: ['slippers', 'stool', 'wire', 'carpet', 'sofa', 'socks', "
                    "'feces', 'table', 'bed', 'closetool', 'book', 'cabinet', 'trashcan', 'curtain']\n"
                )
            elif line.strip().startswith("- "):
                continue
            else:
                f.write(line)

    print(f"✅ YAML 생성 완료: {path}")


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
