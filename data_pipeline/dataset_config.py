# data_pipeline/dataset_config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from data_pipeline.preprocess import copy_and_prune_dataset, copy_augmented_files
from data_pipeline.preprocess_tirod import (
    copy_and_prune_dataset_tirod, copy_augmented_files_tirod
)
from data_pipeline.config_utils import create_yaml, create_tirod_yaml


BASE = Path("/home") / Path().home().name / "jh_ws/yolo"

@dataclass
class DatasetConfig:
    key: str
    name: str                  # datasets/<name>_
    output_prefix: str         # output/<prefix>_
    default_yolo_script: str
    create_yaml_fn: Callable
    copy_prune_fn: Callable
    copy_aug_fn: Callable
    supports_interval: bool = False
    label_fallback_zero: bool = False

CONFIGS = {
    "odsr": DatasetConfig(
        key="odsr",
        name="ODSR-IHS",
        output_prefix="ODSR",
        default_yolo_script=str(BASE / "yolo_train_ODSR_half.py"),
        create_yaml_fn=create_yaml,
        copy_prune_fn=copy_and_prune_dataset,
        copy_aug_fn=copy_augmented_files,
    ),
    "tirod": DatasetConfig(
        key="tirod",
        name="TiROD",
        output_prefix="TiROD",
        default_yolo_script=str(BASE / "yolo_train_TiROD.py"),
        create_yaml_fn=create_tirod_yaml,
        copy_prune_fn=copy_and_prune_dataset_tirod,
        copy_aug_fn=copy_augmented_files_tirod,
        supports_interval=True,
        label_fallback_zero=True,
    ),
}
