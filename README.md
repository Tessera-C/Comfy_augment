# Comfy Augment: 데이터 증강 파이프라인

## 개요

Comfy Augment는 ComfyUI 프레임워크를 기반으로 생성 모델을 사용하여 합성 데이터를 생성하도록 설계된 강력한 데이터 증강 파이프라인입니다. 데이터 전처리, 이미지 생성 및 품질 평가를 위한 포괄적인 워크플로우를 제공하여 사용자가 고품질 합성 이미지로 데이터셋을 확장할 수 있도록 합니다.

이 프로젝트는 특히 강력한 모델을 훈련시키기 위해 대량의 다양한 데이터가 필요한 컴퓨터 비전 작업에 유용합니다.

## 주요 기능

- **데이터 전처리:** 생성 모델을 위한 입력 데이터를 준비하는 전용 파이프라인 (`data_pipeline/preprocess.py`).
- **자동화된 생성:** 구성 가능한 워크플로우를 사용하여 이미지 생성 프로세스를 자동화하는 스크립트 (`run_pipeline.py`).
- **품질 평가:** 생성된 이미지의 품질을 측정하는 도구 (예: LPIPS, DreamSim, FID).
- **일괄 처리:** 대량의 데이터를 효율적으로 처리하기 위한 셸 스크립트 (`run_batch.sh`, `lpips_run_batch.sh`).

## 사전 요구 사항

- Python 3.x
- Git

## 설치

1.  **저장소 복제:**
    ```bash
    git clone <repository-url>
    cd Comfy_augment
    ```

2.  **의존성 설치:**
    가상 환경 사용을 권장합니다.
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows에서는 `venv\Scripts\activate` 사용
    ```
    필요한 Python 패키지를 설치합니다:
    ```bash
    pip install -r requirements.txt
    ```

## 사용법

프로젝트의 핵심은 `run_pipeline.py` 스크립트를 통해 실행됩니다. 이 스크립트는 데이터 준비, 학습, 품질 평가 및 결과 복사를 총괄하는 파이프라인입니다.

### 기본 실행 예시

```bash
python run_pipeline.py --version v9,v10 --ratio 0.5 --match-ratio 1.0 --seed 42
```

### 인자 설명

#### 주요 인자

-   `--version`: 사용할 데이터 증강 버전을 지정합니다. 콤마나 공백으로 여러 버전을 지정할 수 있습니다. (예: `v9,v10,v11`)
-   `--dataset`: 사용할 데이터셋 키를 선택합니다. (기본값: `odsr`, 선택 가능: `odsr`, `tirod`)
-   `--ratio`: 원본 데이터셋에서 삭제할 비율을 지정합니다. (기본값: `0.5`)
-   `--match-ratio`: 증강 데이터를 추가할 비율을 지정합니다.
-   `--match-mode`: 증강 데이터 선택 방식을 결정합니다. (기본값: `mix`)
    -   `mix`: 여러 버전을 하나로 합쳐 무작위로 샘플링합니다.
    -   `vs` (verselect): 버전 단위로 전체를 선택합니다. 이 경우 `--match-ratio`는 1 이상의 정수여야 하며, 선택할 버전의 개수를 의미합니다.

#### 시드(Seed) 인자

-   `--seed`: 모든 무작위 과정(삭제, 매칭)에 사용될 공통 시드 값입니다. (기본값: `42`)
-   `--seed-del`: 원본 삭제 시 사용할 시드를 별도로 지정합니다.
-   `--seed-match`: 증강 매칭 시 사용할 시드를 별도로 지정합니다.

#### 동작 제어 인자

-   `--prefix`: 파일명에 버전 접두사를 추가하는 전처리 단계를 실행합니다.
-   `--skip-train`: 데이터셋 구성까지만 실행하고, YOLO 학습은 건너뜁니다.
-   `--save-results`: 학습 성공 시 `results.csv`와 `best.pt`를 `results/` 폴더로 복사합니다.
-   `--quality`: 품질 평가(FID, LPIPS/DreamSim)를 실행하고 로그를 저장합니다.
-   `--quality-only`: 다른 과정 없이 품질 평가만 독립적으로 실행합니다.
-   `--analyze-only`: `runs/detect`에 있는 기존 학습 결과들을 `results/`로 복사만 합니다.

#### 품질 지표 및 필터링 인자

-   `--metric`: 품질 평가에 사용할 지표를 선택합니다. (기본값: `lpips`, 선택 가능: `lpips`, `dreamsim`)
-   `--sampling`: 원본 삭제 샘플링 방식을 선택합니다. (기본값: `random`, `tirod` 데이터셋에서는 `interval` 사용 가능)
-   `--lpips-mode`: LPIPS 점수를 기반으로 데이터를 필터링하는 방식을 선택합니다. (기본값: `range`)
    -   `range`: `--lpips-min`과 `--lpips-max` 사이의 점수를 가진 이미지만 선택합니다.
    -   `top`: 상위 `--lpips-percent`의 이미지만 선택합니다.
    -   `bottom`: 하위 `--lpips-percent`의 이미지만 선택합니다.
    -   `split`: 데이터를 `--lpips-split` 개수로 등분하고, 그중 `--lpips-split-idx`번째 구간만 선택합니다.
-   `--lpips-min`, `--lpips-max`: `range` 모드에서 사용할 최소/최대 LPIPS 값.
-   `--lpips-percent`: `top`/`bottom` 모드에서 사용할 백분율.
-   `--lpips-split`, `--lpips-split-idx`: `split` 모드에서 사용할 등분 개수와 인덱스.

### 고급 실행 예시

#### `verselect` 모드 사용 예시

`v9`, `v10`, `v11` 세 버전 중 2개 버전을 무작위로 선택하여 전체 데이터를 사용하고 싶을 때:

```bash
python run_pipeline.py --version v9,v10,v11 --match-mode vs --match-ratio 2
```

#### LPIPS 점수 기반 필터링 예시

LPIPS 점수가 0.2에서 0.4 사이인 이미지만을 사용하여 데이터셋을 구성하고 학습을 진행할 때:

```bash
python run_pipeline.py --version v12 --ratio 0.3 --lpips-mode range --lpips-min 0.2 --lpips-max 0.4
```

## 프로젝트 구조

```
Comfy_augment/
├───data_pipeline/      # 데이터 전처리 및 품질 지표 스크립트
├───comfy/              # 핵심 ComfyUI 구성 요소
├───comfy_extras/       # ComfyUI를 위한 추가 커스텀 노드
├───models/             # ML 모델 저장 디렉토리 (체크포인트, LoRA 등)
├───input/              # 입력 데이터 기본 디렉토리
├───output/             # 생성된 이미지 및 결과 기본 디렉토리
├───run_pipeline.py     # 주 실행 파이프라인 스크립트
├───requirements.txt    # Python 의존성 목록
└───README.md           # 이 파일
```