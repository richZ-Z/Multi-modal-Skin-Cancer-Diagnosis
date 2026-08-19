"""
Skin Lesion Binary Classification Training
-------------------------------------------
Trains a fusion model to classify skin lesions as Benign (0) or Malignant (1)
using the ISIC metadata.csv as ground truth (diagnosis_1).

Rows where diagnosis_1 is NaN or "Indeterminate" are dropped in dataset.py.

Two-phase training strategy:
  Phase 1 (epochs 1 → UNFREEZE_EPOCH-1): CNN frozen — train clinical + fusion + head
  Phase 2 (epoch UNFREEZE_EPOCH → end):  Unfreeze CNN for joint fine-tuning

Usage:
    python train.py                      # full fusion model
    python train.py --mode vision        # vision-only ablation
    python train.py --mode clinical      # clinical-only ablation
    python train.py --no-wandb           # disable W&B logging
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

import config
from dataset import build_dataloaders
from models import FusionModel

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warn] wandb not installed. Run: pip install wandb")


# ── Device check ──────────────────────────────────────────────────────────────

def check_device() -> torch.device:
    print("\n" + "=" * 55)
    print("DEVICE CHECK")
    print("=" * 55)
    print(f"  PyTorch version : {torch.__version__}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        vram   = props.total_memory / 1024 ** 3
        used   = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(f"  GPU             : {props.name}")
        print(f"  VRAM            : {vram:.1f} GB total  |  {used:.2f} GB used")
        print(f"  CUDA version    : {torch.version.cuda}")
        print(f"  cuDNN version   : {torch.backends.cudnn.version()}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("  [warn] No CUDA GPU detected — running on CPU (training will be slow).")
    print("=" * 55 + "\n")
    return device


# ── Metric helpers ────────────────────────────────────────────────────────────

def _acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def _f1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1).cpu().numpy()
    return f1_score(targets.cpu().numpy(), preds, average="binary", zero_division=0)


# ── One epoch ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = total_correct = n = 0

    for batch_idx, (images, clin, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        clin   = clin.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, clin)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs             = labels.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.detach().argmax(1) == labels).sum().item()
        n             += bs

        if (batch_idx + 1) % config.LOG_INTERVAL == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    return {
        "loss": total_loss    / n,
        "acc":  total_correct / n,
    }


# ── Validation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = n = 0
    all_logits, all_targets = [], []

    for images, clin, labels in loader:
        images = images.to(device, non_blocking=True)
        clin   = clin.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, clin)
        loss   = criterion(logits, labels)

        bs             = labels.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == labels).sum().item()
        n             += bs
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss    / n,
        "acc":  total_correct / n,
        "f1":   _f1(all_logits, all_targets),
    }


# ── Main training loop ────────────────────────────────────────────────────────

def train(mode: str = "fusion", use_wandb: bool = True, epoch_callback=None):
    device = check_device()
    config.DEVICE = device
    print(f"Training mode : {mode}")

    train_loader, val_loader, _, clin_dim = build_dataloaders()

    model = FusionModel(
        clin_input_dim=clin_dim,
        clin_feature_dim=config.CLIN_FEATURE_DIM,
        clin_hidden_dims=config.CLIN_HIDDEN_DIMS,
        fusion_hidden=config.FUSION_HIDDEN_DIMS,
        dropout=config.DROPOUT,
        mode=mode,
    ).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params        : {total:,} total | {trainable:,} trainable (phase-1)\n")

    criterion = nn.CrossEntropyLoss()

    run_is_mine = use_wandb_run = False
    if use_wandb and WANDB_AVAILABLE and config.WANDB_ENABLED:
        if wandb.run is None:
            wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                name=f"{mode}-run",
                config={
                    "mode":             mode,
                    "learning_rate":    config.LEARNING_RATE,
                    "weight_decay":     config.WEIGHT_DECAY,
                    "batch_size":       config.BATCH_SIZE,
                    "epochs":           config.NUM_EPOCHS,
                    "unfreeze_epoch":   config.UNFREEZE_EPOCH,
                    "dropout":          config.DROPOUT,
                    "clin_hidden_dims": config.CLIN_HIDDEN_DIMS,
                    "fusion_hidden":    config.FUSION_HIDDEN_DIMS,
                    "clin_input_dim":   clin_dim,
                    "num_classes":      config.NUM_CLASSES,
                },
            )
            run_is_mine = True
        wandb.watch(model, log="gradients", log_freq=100)
        use_wandb_run = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        if epoch == config.UNFREEZE_EPOCH and mode in ("fusion", "vision"):
            print(f"\nEpoch {epoch}: Phase-2 — unfreezing CNN for joint fine-tuning.")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE * 0.1,
                weight_decay=config.WEIGHT_DECAY,
            )
            scheduler = StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_m   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Loss {val_m['loss']:.4f} | "
            f"Acc {val_m['acc']*100:.1f}%  F1 {val_m['f1']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if use_wandb_run:
            wandb.log({
                "epoch":       epoch,
                "train/loss":  train_m["loss"],
                "train/acc":   train_m["acc"],
                "val/loss":    val_m["loss"],
                "val/acc":     val_m["acc"],
                "val/f1":      val_m["f1"],
                "lr":          scheduler.get_last_lr()[0],
            })

        if epoch_callback is not None:
            epoch_callback(epoch=epoch, val_loss=val_m["loss"],
                           val_acc=val_m["acc"], val_f1=val_m["f1"])

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            path = os.path.join(config.CHECKPOINT_DIR, f"best_{mode}.pt")
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "val_loss":           val_m["loss"],
                "val_acc":            val_m["acc"],
                "val_f1":             val_m["f1"],
                "mode":               mode,
                "clin_input_dim":     clin_dim,
                "clin_feature_dim":   config.CLIN_FEATURE_DIM,
                "clin_hidden_dims":   config.CLIN_HIDDEN_DIMS,
                "fusion_hidden_dims": config.FUSION_HIDDEN_DIMS,
                "dropout":            config.DROPOUT,
            }, path)
            print(f"  -> New best (loss={best_val_loss:.4f}  "
                  f"Acc={val_m['acc']*100:.1f}%  F1={val_m['f1']:.3f})  saved to {path}")
            if use_wandb_run:
                wandb.save(path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    if use_wandb_run and run_is_mine:
        wandb.finish()
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fusion", "vision", "clinical"], default="fusion")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    train(mode=args.mode, use_wandb=not args.no_wandb)
