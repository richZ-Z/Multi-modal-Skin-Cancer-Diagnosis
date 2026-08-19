"""
Central configuration for the Skin Lesion Binary Classification project.
"""

import torch

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_CSV   = "data/metadata.csv"
IMAGE_DIR  = "data/images/dataset"
IMAGE_SIZE = 224   # standard ResNet-50 input size

TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
# TEST_SPLIT  = 0.15 (remainder)

# Columns excluded from MLP (identifiers, copyright, and ground-truth leakage):
#   isic_id, attribution, copyright_license, lesion_id  → identifiers / admin
#   diagnosis_1                                          → ground-truth label
#   diagnosis_2, diagnosis_3                             → sub-categories of label (leakage)

# Categorical columns → will be one-hot encoded
CLIN_CATEGORICAL = [
    "anatom_site_1",
    "anatom_site_2",
    "anatom_site_special",
    "concomitant_biopsy",
    "diagnosis_confirm_type",
    "image_type",
    "melanocytic",
    "sex",
]
# Numerical columns → will be standardized
CLIN_NUMERICAL = ["age_approx"]

# CLIN_INPUT_DIM: actual size of the encoded clinical vector.
# Set dynamically by dataset.build_dataloaders() after one-hot expansion.
CLIN_INPUT_DIM = None

# ── Binary classification target (diagnosis_1) ────────────────────────────────
# Rows where diagnosis_1 is NaN or "Indeterminate" are dropped at load time.
DIAGNOSIS_CLASSES = ["Benign", "Malignant"]   # index 0 / 1
NUM_CLASSES       = 2

# ── Model ─────────────────────────────────────────────────────────────────────
CNN_FEATURE_DIM   = 2048           # ResNet-50 avgpool output size
FREEZE_BACKBONE   = True           # freeze CNN weights during phase-1 training
UNFREEZE_EPOCH    = 10             # epoch at which backbone is unfrozen for joint fine-tuning

CLIN_HIDDEN_DIMS  = [128, 256, 128]  # MLP hidden layer sizes
CLIN_FEATURE_DIM  = 128              # MLP output embedding dimension

FUSION_HIDDEN_DIMS = [512, 256]      # dense layers in the fusion block
DROPOUT            = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
LR_STEP_SIZE  = 10     # StepLR: decay LR every N epochs
LR_GAMMA      = 0.5

# ── I/O ───────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
LOG_INTERVAL   = 20   # print batch loss every N batches

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Weights & Biases ──────────────────────────────────────────────────────────
WANDB_PROJECT = "skin-lesion-classification"
WANDB_ENTITY  = "justin212553"
WANDB_ENABLED = True
