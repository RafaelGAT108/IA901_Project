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

> **Note on disk space:**
> It is recommended to have at least **30 GB** of free disk space to store the raw, interim, and preprocessed data. Additionally, remember to allocate extra space for the model weights.

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

Or, you can use a standard Python virtual environment (`venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on GPU/CUDA compatibility:**
> If you intend to train models on a GPU and the PyTorch version is incompatible with your system's CUDA version, you will need to update or reinstall PyTorch.
> Refer to the [Official PyTorch Start Guide](https://pytorch.org/get-started/locally/) to find the exact installation command for your specific system.

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

- [`lungsound.py`](modules/lungsound.py): load audio files (1D), load features (2D or N-D), and plot waveforms/features.
- [`transforms.py`](modules/transforms.py): audio transforms (resample, pad/trim, etc), feature extractors (STFT, MFCC, etc), and feature transforms (MinMaxNormalization, etc).
- [`datasets.py`](modules/datasets.py): dataset classes, data loading, and train/val/test splits.
- [`datamodule.py`](modules/datamodule.py): pytorch lightning DataModule.
- [`model.py`](modules/model.py): loads a model from torchvision and adapts input/output layers.
- [`classifier.py`](modules/classifier.py): pytorch lightning module for training, validation, and test steps.

## Notebooks

- [`1_data_loading.ipynb`](notebooks/1_data_loading.ipynb): download raw files to data/raw.
- [`2_audio_analysis.ipynb`](notebooks/2_audio_analysis.ipynb): look at class balance, demographics, and audio durations.
- [`3_transforms_analysis.ipynb`](notebooks/3_transforms_analysis.ipynb): perform audio transforms and compare feature extractors.
- [`4_preprocess_audio.ipynb`](notebooks/4_preprocess_audio.ipynb): run the audio preprocessing pipeline and save .wav files to data/interim.
- [`5_preprocess_features.ipynb`](notebooks/5_preprocess_features.ipynb): run the features preprocessing pipeline and save .npz files to data/preprocessed.
- [`6_training.ipynb`](notebooks/7_training.ipynb): run training and evaluate results.