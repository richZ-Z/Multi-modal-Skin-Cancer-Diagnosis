# Multi-Modal Length of Stay Prediction Model

A neural network system for Length of Stay Prediction that fuses **Chest CT images** (CNN) with **clinical data** (MLP) for more robust diagnosis than image-only approaches.

## Architecture

```
Image  →  Vision Branch (ResNet-50)      →  F_img ─┐
                                                   ├→ FusionBlock → Classification Head → Disease Class
Env    →  Clinical Branch (MLP)          →  F_env ─┘
```

Clinical features: Patient_ID, Age, Gender, Medical Condition, Admission Type, Medication

## Setup

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd 274P-Crop-Disease-Detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with the correct CUDA version first — see the comment at the top of `requirements.txt` or visit https://pytorch.org/get-started/locally/

**2. Download the dataset**

Download from Kaggle: [Multi-Modal Healthcare Dataset: Patient Records](https://www.kaggle.com/datasets/ajithdari/multi-modal-healthcare-dataset-patient-records)

Place files as follows:
```
data/
├── dataset.csv        # clinical readings + labels
└── images/            # CT images from various sources
    ├── ChestCT        # Chest CT images
    └── ...
```

<!-- **3. Configure**

Edit [`config.py`](config.py) to match your dataset:
- `ENV_FEATURES` — column names of the environmental features in your CSV
- `CLASS_NAMES` — ordered list of disease class names matching integer labels
- `CNN_BsasdfACKBONE` — `"resnet50"` -->

## Usage

**Train (full fusion model)**
```bash
python train.py
```

**Train a single branch (for ablation)**
```bash
python train.py --mode vision   # CNN only
python train.py --mode env      # MLP only
```

**Evaluate + ablation study**
```bash
python evaluate.py              # all three modes side-by-side
python evaluate.py --mode fusion
```

Confusion matrices and the ablation CSV are saved to `evaluation_results/`.

**Verify data pipeline**
```bash
python dataset.py
```

**Verify model architecture**
```bash
python models/fusion_model.py
```

## Project Structure

```
├── config.py                   # hyperparameters and feature definitions
├── dataset.py                  # P1 — data loading, augmentation, preprocessing
├── models/
│   ├── vision_branch.py        # P2 — CNN feature extractor
│   ├── clinical_branch.py           # P3 — MLP feature extractor
│   └── fusion_model.py         # P4 — fusion architecture
├── train.py                    # P4 — end-to-end training loop
├── evaluate.py                 # P5 — metrics, confusion matrix, ablation study
├── data/                       # dataset goes here (not committed)
└── checkpoints/                # saved model weights (not committed)
```

