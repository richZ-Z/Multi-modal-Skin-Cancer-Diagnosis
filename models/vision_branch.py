"""P2 – Vision Branch: ImageNet pre-trained ResNet-50 used as a feature extractor."""

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config


class VisionBranch(nn.Module):
    """
    ResNet-50 backbone with the final FC layer removed.
    Outputs a visual context vector F_img of shape (B, 2048).

    Training strategy (from design doc):
      Phase 1 — backbone frozen:  only clinical branch + fusion head are trained.
      Phase 2 — call unfreeze():  joint fine-tuning of the entire network.
    """

    def __init__(self, freeze: bool = config.FREEZE_BACKBONE):
        super().__init__()
        backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the classification head; keep conv layers + global avg-pool
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def unfreeze(self):
        """Enable joint fine-tuning after UNFREEZE_EPOCH."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(start_dim=1)   # (B, 2048)
