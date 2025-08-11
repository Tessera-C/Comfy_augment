# Comfy Augment: Data Augmentation Pipeline

## Overview

Comfy Augment is a powerful data augmentation pipeline designed to generate synthetic data using generative models, based on the ComfyUI framework. It provides a comprehensive workflow for data preprocessing, image generation, and quality assessment, enabling users to expand their datasets with high-quality synthetic images.

This project is particularly useful for computer vision tasks where large amounts of varied data are required for training robust models.

## Features

- **Data Preprocessing:** A dedicated pipeline (`data_pipeline/preprocess.py`) to prepare input data for the generation models.
- **Automated Generation:** A script (`run_pipeline.py`) to automate the image generation process using configurable workflows.
- **Quality Assessment:** Tools to measure the quality of generated images (e.g., using LPIPS, DreamSim, FID).
- **Batch Processing:** Shell scripts (`run_batch.sh`, `lpips_run_batch.sh`) for processing large amounts of data efficiently.

## Prerequisites

- Python 3.x
- Git

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Comfy_augment
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The core of the project is executed through the `run_pipeline.py` script. This script is a pipeline that manages data preparation, training, quality assessment, and result copying.

### Basic Usage Example

```bash
python run_pipeline.py --version v9,v10 --ratio 0.5 --match-ratio 1.0 --seed 42
```

### Argument Descriptions

#### Main Arguments

-   `--version`: Specify the data augmentation versions to use. Multiple versions can be specified with commas or spaces. (e.g., `v9,v10,v11`)
-   `--dataset`: Select the dataset key to use. (default: `odsr`, choices: `odsr`, `tirod`)
-   `--ratio`: Specify the ratio of the original dataset to delete. (default: `0.5`)
-   `--match-ratio`: Specify the ratio of augmented data to add.
-   `--match-mode`: Determine the augmented data selection method. (default: `mix`)
    -   `mix`: Mix multiple versions and sample randomly.
    -   `vs` (verselect): Select entire versions. In this case, `--match-ratio` must be an integer greater than or equal to 1, representing the number of versions to select.

#### Seed Arguments

-   `--seed`: Common seed value to be used for all random processes (deletion, matching). (default: `42`)
-   `--seed-del`: Specify a separate seed for original data deletion.
-   `--seed-match`: Specify a separate seed for augmentation matching.

#### Operation Control Arguments

-   `--prefix`: Execute the preprocessing step to add version prefixes to filenames.
-   `--skip-train`: Run only up to dataset creation and skip YOLO training.
-   `--save-results`: Copy `results.csv` and `best.pt` to the `results/` folder upon successful training.
-   `--quality`: Run quality assessment (FID, LPIPS/DreamSim) and save the logs.
-   `--quality-only`: Run only the quality assessment independently.
-   `--analyze-only`: Only copy existing training results from `runs/detect` to `results/`.

#### Quality Metrics and Filtering Arguments

-   `--metric`: Select the metric to use for quality assessment. (default: `lpips`, choices: `lpips`, `dreamsim`)
-   `--sampling`: Select the original data deletion sampling method. (default: `random`, `interval` can be used for the `tirod` dataset)
-   `--lpips-mode`: Select the method for filtering data based on LPIPS scores. (default: `range`)
    -   `range`: Select only images with scores between `--lpips-min` and `--lpips-max`.
    -   `top`: Select only the top `--lpips-percent` of images.
    -   `bottom`: Select only the bottom `--lpips-percent` of images.
    -   `split`: Divide the data into `--lpips-split` parts and select the `--lpips-split-idx`-th section.
-   `--lpips-min`, `--lpips-max`: Min/max LPIPS values to use in `range` mode.
-   `--lpips-percent`: Percentage to use in `top`/`bottom` mode.
-   `--lpips-split`, `--lpips-split-idx`: Number of splits and index to use in `split` mode.

### Advanced Usage Examples

#### Using `verselect` Mode

To randomly select 2 out of 3 versions (`v9`, `v10`, `v11`) and use all their data:

```bash
python run_pipeline.py --version v9,v10,v11 --match-mode vs --match-ratio 2
```

#### LPIPS Score-based Filtering Example

To create a dataset and train using only images with LPIPS scores between 0.2 and 0.4:

```bash
python run_pipeline.py --version v12 --ratio 0.3 --lpips-mode range --lpips-min 0.2 --lpips-max 0.4
```

## Project Structure

```
Comfy_augment/
├───data_pipeline/      # Scripts for data preprocessing and quality metrics
├───comfy/              # Core ComfyUI components
├───comfy_extras/       # Additional custom nodes for ComfyUI
├───models/             # Directory for storing ML models (checkpoints, LoRAs, etc.)
├───input/              # Default directory for input data
├───output/             # Default directory for generated images and results
├───run_pipeline.py     # Main script for the augmentation pipeline
├───requirements.txt    # List of Python dependencies
└───README.md           # This file
```
