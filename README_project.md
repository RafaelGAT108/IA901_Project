# IA901 Project — Lung Disease Classification

## Team

- Letícia Lopes Mendes Da Silva (RA: 184423)
- Rafael Ávila dos Santos (RA: 300905)
- Sofia Ballerini de Vasconcellos (RA: 299904)

## Overview

This project focuses on **lung disease classification from respiratory audio signals**.

We use signal processing techniques to generate different audio representations (e.g., spectrograms, MFCCs) that support analysis and downstream modeling.

## Datasets

This repository uses public respiratory sound datasets and stores derived features for experiments and analysis.

- ICBHI Challenge (2017): <https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge>
- KAUH (2021): <https://data.mendeley.com/datasets/jwyy9np4gv/3>

See [data/datasheet.pdf](data/datasheet.pdf) for more details.

## Python environment

### Conda

Create and activate the Conda environment from `environment/conda.yaml`:

```bash
conda env create -f environment/conda.yaml
conda activate lung_sounds
```

To update:

```bash
conda env update -f environment/conda.yaml --prune
```

### Venv

Alternatively, you can use a standard Python virtual environment (`venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Structure

```tree
.
├── assets
│   ├── preliminary_results
│   └── workflow.png
├── checkpoints
├── data
│   ├── datasheet.pdf
│   ├── interim
│   ├── preprocessed
│   └── raw
├── environment
│   ├── conda.yaml
│   └── requirements.txt
├── experiments
│   └── template.yaml
├── logs
├── modules
│   ├── classifier.py
│   ├── datamodule.py
│   ├── datasets.py
│   ├── lungsound.py
│   ├── model.py
│   └── transforms.py
├── notebooks
│   ├── 1_data_loading.ipynb
│   ├── 2_audio_analysis.ipynb
│   ├── 3_transforms_analysis.ipynb
│   ├── 4_preprocess.ipynb
│   ├── 5_features_analysis.ipynb
│   ├── 6_datamodule_analysis.ipynb
│   └── 7_training.ipynb
├── results
├── .gitignore
├── LICENSE
├── README.md
└── README_project.md
```

- `assets/`: figures, diagrams, sheets, and visual outputs used in documentation.
- `checkpoints/`: model checkpoints saved during training.
- `data/`: raw, intermediate and preprocessed data, and datasets documentation.
- `environment/`: setup for python environment.
- `experiments/`: training and evaluation configurations for different model runs.
- `logs/`: training logs and experiment tracking outputs.
- `modules/`: core Python modules for data handling, transforms, models, and training.
- `notebooks/`: exploratory analysis and pipeline walkthroughs.
- `results/`: results of model inference on the test set.

## Modules

- [`lungsound.py`](modules/lungsound.py): load audio files (1D), load features (2D), and plot waveforms/features.
- [`transforms.py`](modules/transforms.py): audio transforms (resample, pad/trim, etc.) and feature extractors (STFT, MFCC, etc.).
- [`datasets.py`](modules/datasets.py): dataset classes, label mapping, and train/val/test splits.
- [`datamodule.py`](modules/datamodule.py): pytorch lightning data module with dataloaders, samplers, and collate logic.
- [`model.py`](modules/model.py): loads a model from torchvision and adapts input/output layers.
- [`classifier.py`](modules/classifier.py): pytorch lightning module for training, validation, and test steps.

## Notebooks

- [`1_data_loading.ipynb`](notebooks/1_data_loading.ipynb): download data and check the raw files.
- [`2_audio_analysis.ipynb`](notebooks/2_audio_analysis.ipynb): look at class balance, demographics, and audio durations.
- [`3_transforms_analysis.ipynb`](notebooks/3_transforms_analysis.ipynb): perform audio transforms and compare feature extractors.
- [`4_preprocess.ipynb`](notebooks/4_preprocess.ipynb): run the preprocessing pipeline and save features.
- [`5_features_analysis.ipynb`](notebooks/5_features_analysis.ipynb): inspect extracted features and sample outputs.
- [`6_datamodule_analysis.ipynb`](notebooks/6_datamodule_analysis.ipynb): test dataloaders and batches.
- [`7_training.ipynb`](notebooks/7_training.ipynb): run training and evaluate results.
