"""
Data pipeline for ISIC skin lesion binary classification.

Target: diagnosis_1 from metadata.csv
  - Benign    → 0
  - Malignant → 1
  - NaN / Indeterminate → dropped

Excluded columns (identifiers / copyright / label leakage):
  isic_id, attribution, copyright_license, lesion_id,
  diagnosis_1 (label), diagnosis_2, diagnosis_3

Clinical features fed to the MLP:
  Numerical  : age_approx
  Categorical: anatom_site_1, anatom_site_2, anatom_site_special,
               concomitant_biopsy, diagnosis_confirm_type,
               image_type, melanocytic, sex
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


DIAGNOSIS_MAP = {"Benign": 0, "Malignant": 1}


def get_image_transforms(split: str) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if split == "train":
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32)),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class SkinLesionDataset(Dataset):
    """
    Returns (image, clinical_vector, label) per sample.

    image           — (3, H, W) float32 tensor, ResNet-normalized
    clinical_vector — (clin_input_dim,) float32, one-hot + standardized
    label           — int64 scalar: 0=Benign, 1=Malignant
    """

    def __init__(
        self,
        df:        pd.DataFrame,
        clin_cols: list,
        image_dir: str,
        transform=None,
    ):
        self.df        = df.reset_index(drop=True)
        self.clin_cols = clin_cols
        self.image_dir = image_dir
        self.transform = transform
        self._black    = Image.new("RGB", (config.IMAGE_SIZE + 32, config.IMAGE_SIZE + 32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, f"{row['isic_id']}.jpg")
        try:
            image = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            image = self._black
        if self.transform:
            image = self.transform(image)

        clin = torch.tensor(
            row[self.clin_cols].values.astype(np.float32),
            dtype=torch.float32,
        )

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return image, clin, label


def build_dataloaders(
    csv_path:    str = config.DATA_CSV,
    image_dir:   str = config.IMAGE_DIR,
    batch_size:  int = config.BATCH_SIZE,
    num_workers: int = 0,
):
    """
    Reads metadata.csv, filters rows, encodes features, splits, and returns
    (train_loader, val_loader, test_loader, clin_input_dim).
    """
    df = pd.read_csv(csv_path)

    # ── Drop rows without a usable diagnosis_1 label ──────────────────────────
    before = len(df)
    df = df[df["diagnosis_1"].notna()]
    df = df[df["diagnosis_1"] != "Indeterminate"]
    dropped = before - len(df)
    if dropped:
        print(f"[info] Dropped {dropped} rows (NaN or Indeterminate diagnosis_1).")

    # ── Binary label ──────────────────────────────────────────────────────────
    df = df.copy()
    df["label"] = df["diagnosis_1"].map(DIAGNOSIS_MAP)

    # ── One-hot encode categorical clinical features ───────────────────────────
    cat_dummies = pd.get_dummies(df[config.CLIN_CATEGORICAL], dtype=float)
    cat_cols    = cat_dummies.columns.tolist()

    df_proc = pd.concat(
        [df[["isic_id", "label"] + config.CLIN_NUMERICAL], cat_dummies],
        axis=1,
    )
    clin_cols = config.CLIN_NUMERICAL + cat_cols

    # ── Stratified train / val / test split ───────────────────────────────────
    train_df, temp_df = train_test_split(
        df_proc,
        test_size=(1 - config.TRAIN_SPLIT),
        random_state=42,
        stratify=df_proc["label"],
    )
    val_size_rel = config.VAL_SPLIT / (1 - config.TRAIN_SPLIT)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size_rel),
        random_state=42,
        stratify=temp_df["label"],
    )

    # ── Standardize numerical features (fit on train only) ────────────────────
    scaler = StandardScaler()
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    # Fill NaN in numerical columns with training median before scaling
    num_medians = train_df[config.CLIN_NUMERICAL].median()
    train_df[config.CLIN_NUMERICAL] = train_df[config.CLIN_NUMERICAL].fillna(num_medians)
    val_df[config.CLIN_NUMERICAL]   = val_df[config.CLIN_NUMERICAL].fillna(num_medians)
    test_df[config.CLIN_NUMERICAL]  = test_df[config.CLIN_NUMERICAL].fillna(num_medians)

    train_df[config.CLIN_NUMERICAL] = scaler.fit_transform(train_df[config.CLIN_NUMERICAL])
    val_df[config.CLIN_NUMERICAL]   = scaler.transform(val_df[config.CLIN_NUMERICAL])
    test_df[config.CLIN_NUMERICAL]  = scaler.transform(test_df[config.CLIN_NUMERICAL])

    clin_input_dim = len(clin_cols)
    config.CLIN_INPUT_DIM = clin_input_dim

    counts = df_proc["label"].value_counts().sort_index()
    print(f"Split  - train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    print(f"Clinical vector size: {clin_input_dim}")
    print("Label distribution:")
    for idx, count in counts.items():
        name = config.DIAGNOSIS_CLASSES[idx]
        print(f"  {name:12s}: {count:6,}  ({count/len(df_proc)*100:.1f}%)")

    shared = dict(clin_cols=clin_cols, image_dir=image_dir)
    train_ds = SkinLesionDataset(train_df, **shared, transform=get_image_transforms("train"))
    val_ds   = SkinLesionDataset(val_df,   **shared, transform=get_image_transforms("val"))
    test_ds  = SkinLesionDataset(test_df,  **shared, transform=get_image_transforms("test"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, clin_input_dim


if __name__ == "__main__":
    train_loader, val_loader, test_loader, clin_dim = build_dataloaders()
    images, clin, labels = next(iter(train_loader))
    print(f"\nImage batch     : {images.shape}")
    print(f"Clinical vector : {clin.shape}")
    print(f"Label samples   : {labels[:8].tolist()}")
    print(f"  ({[config.DIAGNOSIS_CLASSES[l] for l in labels[:8].tolist()]})")
