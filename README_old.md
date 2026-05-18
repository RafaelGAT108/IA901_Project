# IA901 Project — Lung Disease Classification

## Team
- Rafael Ávila dos Santos (RA: 300905)  
- Letícia Lopes Mendes Da Silva (RA: 184423)  
- Sofia Ballerini de Vasconcellos	(RA: 299904)  

---

## Overview
This project focuses on **lung disease classification from respiratory audio signals**.  
We use signal processing techniques to generate different audio representations (e.g., spectrograms, MFCCs) that support analysis and downstream modeling.

The dataset used is the well-known **ICBHI Respiratory Sound Database**.

---

## Dataset

Download the dataset from the official source:

🔗 https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip

After downloading, extract it and organize the directory as follows:

```bash
project_root/
│
├── ICBHI_final_database/
│   ├── audio_and_txt_files
│
├── jsons/
│   ├── jsons_file
│
├── notebooks/
│   ├── audio_analysis.ipynb
│   └── pre_processing.ipynb
│
└── README_old.md
```


---


### Audio Analysis

Run the notebook:

```bash
notebooks/audio_analysis.ipynb
```

This notebook allows you to:

[//]: # (- Visualize respiratory signals)
- Compare spectrograms across different lung diseases

[//]: # (- Perform exploratory signal analysis)

---

### 3. Pre-processing Pipeline

Run:


```bash
notebooks/pre_processing.ipynb
```


This step will:

- Process all audio samples in the dataset, grouping  in specific folder for each disease.
- Generate multiple feature representations, including:
  - Mel Spectrograms  
  - MFCC (Mel-Frequency Cepstral Coefficients)  
  - STFT (Short-Time Fourier Transform)  

- Organize outputs into structured folders by representation type

---

## Output Structure

After preprocessing, the data will be organized like:

```bash
processed_data_ICBHI/
├── mel_spectrogram/
├── mfcc/
├── stft/
```
