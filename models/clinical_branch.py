"""P3 – Clinical Branch: MLP over one-hot encoded + standardized patient metadata."""

import torch
import torch.nn as nn

import config


class ClinicalBranch(nn.Module):
    """
    MLP that maps the clinical context vector to a dense embedding F_clin.

    input_dim is passed explicitly from the caller (set by build_dataloaders
    after one-hot expansion). Typical size for the Kaggle dataset: ~25.

    Architecture per design doc:
        Linear → BN → ReLU → Dropout   (repeated for each hidden layer)
        Linear → ReLU                   (output embedding)
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: list  = config.CLIN_HIDDEN_DIMS,
        output_dim:  int   = config.CLIN_FEATURE_DIM,
        dropout:     float = config.DROPOUT,
    ):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.ReLU(inplace=True)]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)   # (B, CLIN_FEATURE_DIM)
