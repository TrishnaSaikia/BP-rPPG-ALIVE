from __future__ import annotations

import torch
from torch import nn


class NegativePearsonLoss(nn.Module):
    """Paper feature-alignment loss L_F = negative Pearson correlation."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        if pred.shape != target.shape:
            raise ValueError(
                f"Feature shapes must match for L_F: {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        numerator = (pred * target).sum(dim=1)
        denominator = torch.sqrt((pred.square()).sum(dim=1) + self.eps) * torch.sqrt(
            (target.square()).sum(dim=1) + self.eps
        )
        return -(numerator / denominator).mean()


class DataFidelityLoss(nn.Module):
    """Paper data-fidelity loss L_DF for jointly estimating SBP and DBP."""

    def forward(self, pred_bp: torch.Tensor, true_bp: torch.Tensor) -> torch.Tensor:
        if pred_bp.shape != true_bp.shape or pred_bp.shape[-1] != 2:
            raise ValueError(
                f"Expected matching [B, 2] BP tensors, got {pred_bp.shape} and {true_bp.shape}"
            )
        return (pred_bp - true_bp).square().sum(dim=1).mean()
