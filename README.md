# Comfy Augment

This project is designed to augment image datasets and train YOLO models. The core logic is encapsulated in the `run_pipeline.py` script, which automates the entire workflow from data preparation to model training and evaluation.

## `run_pipeline.py`

This script is the main entry point for the project. It provides a command-line interface to run a complete pipeline for preparing YOLO (You Only Look Once) datasets, training models, and evaluating their quality.

### Features

*   **Data Preparation:**
    *   Copies and prunes datasets based on specified ratios.
    *   Copies augmented files based on a matching ratio.
    *   Adds prefixes to filenames for versioning.
*   **Quality Evaluation:**
    *   Computes FID (Fréchet Inception Distance) to measure the quality of generated images.
    *   Computes LPIPS (Learned Perceptual Image Patch Similarity) to measure the similarity between images.
    *   Filters the dataset based on LPIPS scores.
*   **Training:**
    *   Creates a YAML configuration file for YOLO training.
    *   Runs the YOLO training script.
    *   Saves the training results.

### Usage

```bash
python run_pipeline.py --version <versions> [options]
```

*   `--version`: Specify the versions of the dataset to use.
*   `--ratio`: The ratio of the original dataset to delete.
*   `--match-ratio`: The ratio of augmented files to match.
*   `--seed`: The random seed for reproducibility.
*   `--prefix`: Add a prefix to filenames.
*   `--skip-train`: Skip the training step.
*   `--save-results`: Save the training results.
*   `--quality`: Run the quality evaluation.
*   `--quality-only`: Only run the quality evaluation.
*   `--analyze-only`: Only analyze the results.
*   `--lpips-mode`: The LPIPS filter mode (`range`, `top`, `bottom`, `split`).
*   `--lpips-min`, `--lpips-max`, `--lpips-percent`, `--lpips-split`, `--lpips-split-idx`: Parameters for the LPIPS filter.

## Other Scripts

The other Python scripts in this project provide additional features and utilities that support the main pipeline.