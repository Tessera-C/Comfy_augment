# Comfy Augment: Data Augmentation Pipeline

## Overview

Comfy Augment is a powerful data augmentation pipeline designed to generate synthetic data using generative models, based on the ComfyUI framework. It provides a comprehensive workflow for data preprocessing, image generation, and quality assessment, enabling users to expand their datasets with high-quality synthetic images.

This project is particularly useful for tasks in computer vision where large amounts of varied data are required for training robust models.

## Features

- **Data Preprocessing:** A dedicated pipeline (`data_pipeline/preprocess.py`) to prepare input data for the generation models.
- **Automated Generation:** Scripts (`run_pipeline.py`, `run_pipeline_tirod.py`) to automate the image generation process using configurable workflows.
- **TiROD Integration:** Specialized pipeline for running the TiROD model for object detection tasks.
- **Quality Assessment:** Tools to measure the quality of generated images (e.g., using LPIPS).
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

The core of the project is executed through the `run_pipeline.py` and `run_pipeline_tirod.py` scripts.

### Standard Pipeline

The `run_pipeline.py` script orchestrates the main data augmentation workflow. You can run it with various arguments to control the pipeline's behavior.

**Example:**
```bash
python run_pipeline.py --config_path /path/to/your/config.json --input_dir /path/to/input/images --output_dir /path/to/output
```
*(Note: Please check the script's argument parser for a full list of available options.)*

### TiROD Pipeline

The `run_pipeline_tirod.py` script is specifically designed for workflows involving the TiROD model.

**Example:**
```bash
python run_pipeline_tirod.py --config_path /path/to/your/tirod_config.json --input_dir /path/to/input/images --output_dir /path/to/output
```

### Batch Processing

For generating data from multiple input directories or with different configurations, you can use the provided shell scripts.

-   `run_batch.sh`: A script for running the pipeline on multiple data batches.
-   `lpips_run_batch.sh`: A script for running LPIPS quality assessment on batches of generated images.

Modify these scripts as needed to fit your specific workflow.

## Project Structure

```
Comfy_augment/
├───data_pipeline/      # Scripts for data preprocessing and quality metrics
├───comfy/              # Core ComfyUI components
├───comfy_extras/       # Additional custom nodes for ComfyUI
├───models/             # Directory for storing ML models (checkpoints, LoRAs, etc.)
├───input/              # Default directory for input data
├───output/             # Default directory for generated images and results
├───run_pipeline.py     # Main script for the standard augmentation pipeline
├───run_pipeline_tirod.py # Main script for the TiROD pipeline
├───requirements.txt    # List of Python dependencies
└───README.md           # This file
```

## Key Dependencies

This project relies on several key libraries, including:
- `torch` & `torchvision`
- `diffusers`
- `transformers`
- `Pillow`
- `lpips`

Please refer to `requirements.txt` for the full list.
