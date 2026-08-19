"""Fusion model: VisionBranch + ClinicalBranch → binary Benign/Malignant classifier."""

import torch
import torch.nn as nn

import config
from models.vision_branch   import VisionBranch
from models.clinical_branch import ClinicalBranch


class FusionModel(nn.Module):
    """
    Binary skin-lesion classifier.

        VisionBranch    →  F_img  (B, CNN_FEATURE_DIM)
        ClinicalBranch  →  F_clin (B, CLIN_FEATURE_DIM)
        concat          →  Shared FusionBlock  →  classifier head
                                                   (B, 2)  Benign / Malignant

    forward() returns raw logits of shape (B, 2).

    Ablation modes: "fusion" | "vision" | "clinical"
    """

    def __init__(
        self,
        clin_input_dim:   int,
        cnn_feature_dim:  int   = config.CNN_FEATURE_DIM,
        clin_feature_dim: int   = config.CLIN_FEATURE_DIM,
        clin_hidden_dims: list  = None,
        fusion_hidden:    list  = None,
        dropout:          float = config.DROPOUT,
        num_classes:      int   = config.NUM_CLASSES,
        mode:             str   = "fusion",
    ):
        super().__init__()
        assert mode in ("fusion", "vision", "clinical"), f"Unknown mode: {mode}"
        self.mode = mode

        # Fall back to config values at call time (not at import/definition time)
        if clin_hidden_dims is None:
            clin_hidden_dims = config.CLIN_HIDDEN_DIMS
        if fusion_hidden is None:
            fusion_hidden = config.FUSION_HIDDEN_DIMS

        self.vision_branch   = VisionBranch()
        self.clinical_branch = ClinicalBranch(
            input_dim=clin_input_dim,
            hidden_dims=clin_hidden_dims,
            output_dim=clin_feature_dim,
            dropout=dropout,
        )

        if mode == "fusion":
            trunk_input = cnn_feature_dim + clin_feature_dim
        elif mode == "vision":
            trunk_input = cnn_feature_dim
        else:
            trunk_input = clin_feature_dim

        trunk_layers = []
        prev = trunk_input
        for h in fusion_hidden:
            trunk_layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        self.fusion_trunk = nn.Sequential(*trunk_layers)

        self.classifier = nn.Linear(prev, num_classes)

    def unfreeze_backbone(self):
        self.vision_branch.unfreeze()

    def forward(self, image: torch.Tensor, clin: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, 2)."""
        if self.mode == "fusion":
            fused = torch.cat([self.vision_branch(image), self.clinical_branch(clin)], dim=1)
        elif self.mode == "vision":
            fused = self.vision_branch(image)
        else:
            fused = self.clinical_branch(clin)

        return self.classifier(self.fusion_trunk(fused))


if __name__ == "__main__":
    from dataset import build_dataloaders
    train_loader, _, _, clin_dim = build_dataloaders()

    device = config.DEVICE
    model  = FusionModel(clin_input_dim=clin_dim, mode="fusion",
                         clin_hidden_dims=config.CLIN_HIDDEN_DIMS,
                         clin_feature_dim=config.CLIN_FEATURE_DIM,
                         fusion_hidden=config.FUSION_HIDDEN_DIMS,
                         dropout=config.DROPOUT).to(device)

    images, clin, labels = next(iter(train_loader))
    images = images.to(device)
    clin   = clin.to(device)
    labels = labels.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(images, clin)

    print(f"Batch size  : {images.shape[0]}")
    print(f"Logits      : {tuple(logits.shape)}")
    print(f"Predictions : {logits.argmax(1)[:8].tolist()}")
    print(f"Labels      : {labels[:8].tolist()}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable (phase-1): {trainable:,}")
