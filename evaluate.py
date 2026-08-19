"""
Skin Lesion Binary Classification – Evaluation & Ablation Study
---------------------------------------------------------------
Reports test-set performance for each checkpoint:

  Binary Classification (Benign / Malignant):
      Accuracy, weighted F1, per-class precision/recall/F1
      Confusion matrix

  Ablation table compares all metrics across vision-only,
  clinical-only, and full fusion models.

Usage:
    python evaluate.py                     # ablation across all modes
    python evaluate.py --mode fusion       # evaluate best_fusion.pt only
    python evaluate.py --mode vision       # evaluate best_vision.pt only
    python evaluate.py --mode clinical     # evaluate best_clinical.pt only
    python evaluate.py --no-wandb
"""

import os
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import config
from dataset import build_dataloaders
from models import FusionModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_inference(model, loader, device):
    """Returns (preds, targets) as numpy arrays of integer class indices."""
    model.eval()
    all_preds, all_targets = [], []

    for images, clin, labels in loader:
        images = images.to(device, non_blocking=True)
        clin   = clin.to(device, non_blocking=True)

        logits = model(images, clin)
        preds  = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, targets):
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1":       f1_score(targets, preds, average="weighted", zero_division=0),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion(preds, targets, mode):
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.DIAGNOSIS_CLASSES,
        yticklabels=config.DIAGNOSIS_CLASSES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {mode}")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_{mode}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── Single-mode evaluation ────────────────────────────────────────────────────

def evaluate_mode(mode, test_loader, clin_dim, device, use_wandb):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_{mode}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [skip] No checkpoint found for mode='{mode}' at {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)

    model = FusionModel(
        clin_input_dim=ckpt.get("clin_input_dim", clin_dim),
        clin_feature_dim=ckpt.get("clin_feature_dim", config.CLIN_FEATURE_DIM),
        clin_hidden_dims=ckpt.get("clin_hidden_dims", config.CLIN_HIDDEN_DIMS),
        fusion_hidden=ckpt.get("fusion_hidden_dims", config.FUSION_HIDDEN_DIMS),
        dropout=ckpt.get("dropout", config.DROPOUT),
        mode=mode,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_loss = ckpt.get("val_loss", float("nan"))
    val_acc  = ckpt.get("val_acc",  float("nan"))
    val_f1   = ckpt.get("val_f1",   float("nan"))

    print(f"\n{'='*55}")
    print(f"Evaluating: {mode}")
    print(f"  val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%  val_f1={val_f1:.3f}")
    print(f"{'='*55}")

    preds, targets = run_inference(model, test_loader, device)
    m = compute_metrics(preds, targets)

    print(f"\n[Binary Classification — Benign / Malignant]")
    print(f"  Accuracy : {m['accuracy']*100:.2f}%")
    print(f"  F1 (wtd) : {m['f1']:.4f}")
    print("\n  Per-class report:")
    print(classification_report(
        targets, preds,
        target_names=config.DIAGNOSIS_CLASSES,
        zero_division=0,
    ))

    conf_plot = plot_confusion(preds, targets, mode)
    print(f"  Plot saved -> {conf_plot}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            f"{mode}/test_accuracy":    m["accuracy"],
            f"{mode}/test_f1":          m["f1"],
            f"{mode}/confusion_matrix": wandb.Image(conf_plot),
        })

    return m


# ── Ablation study ────────────────────────────────────────────────────────────

def ablation_study(test_loader, clin_dim, device, use_wandb):
    modes   = ["vision", "clinical", "fusion"]
    results = {}
    for mode in modes:
        m = evaluate_mode(mode, test_loader, clin_dim, device, use_wandb)
        if m:
            results[mode] = m

    if len(results) < 2:
        print("\nNot enough checkpoints for a full ablation comparison.")
        return

    print(f"\n{'='*55}")
    print("ABLATION STUDY — Benign / Malignant Classification")
    print(f"{'='*55}")
    df = pd.DataFrame(results).T.rename(columns={
        "accuracy": "Accuracy",
        "f1":       "F1 (weighted)",
    })
    print(df.to_string(float_format="{:.4f}".format))

    csv_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df.to_csv(csv_path)
    print(f"\nAblation table saved -> {csv_path}")

    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"ablation_table": wandb.Table(
            dataframe=df.reset_index().rename(columns={"index": "mode"})
        )})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fusion", "vision", "clinical", "all"],
        default="all",
        help="Which checkpoint to evaluate. 'all' runs the full ablation study.",
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    use_wandb = not args.no_wandb
    device    = config.DEVICE

    _, _, test_loader, clin_dim = build_dataloaders()

    if use_wandb and WANDB_AVAILABLE and config.WANDB_ENABLED:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=f"eval-{args.mode}",
            job_type="eval",
        )

    if args.mode == "all":
        ablation_study(test_loader, clin_dim, device, use_wandb)
    else:
        evaluate_mode(args.mode, test_loader, clin_dim, device, use_wandb)

    if use_wandb and WANDB_AVAILABLE and config.WANDB_ENABLED and wandb.run:
        wandb.finish()
