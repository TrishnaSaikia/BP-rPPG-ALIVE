"""Teacher and student networks used by ALIVE."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .mtcn import PaperMTCN


def _make_regression_head(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 2,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        if hidden <= 0:
            raise ValueError("All MLP hidden dimensions must be positive")
        layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class BPRegressor(nn.Module):
    """Shared M-TCN plus MLP BP regression implementation."""

    def __init__(
        self,
        input_rows: int,
        clip_samples: int = 120,
        internal_filters: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.01,
        dilation_powers: Sequence[int] | None = None,
        mlp_hidden: Sequence[int] = (64,),
    ) -> None:
        super().__init__()
        self.input_rows = input_rows
        self.clip_samples = clip_samples
        self.backbone = PaperMTCN(
            input_rows=input_rows,
            clip_samples=clip_samples,
            internal_filters=internal_filters,
            kernel_size=kernel_size,
            dropout=dropout,
            dilation_powers=dilation_powers,
        )
        self.bp_head = _make_regression_head(clip_samples, mlp_hidden, output_dim=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        bp = self.bp_head(features)
        return features, bp

    def model_config(self) -> dict:
        first_block = self.backbone.blocks[0]
        first_conv = first_block.conv1[0]
        dropout_layer = first_block.conv1[3]
        hidden_dims = [
            module.out_features
            for module in self.bp_head
            if isinstance(module, nn.Linear)
        ][:-1]
        return {
            "input_rows": self.input_rows,
            "clip_samples": self.clip_samples,
            "internal_filters": first_conv.out_channels,
            "kernel_size": first_conv.kernel_size[1],
            "dropout": dropout_layer.p,
            "dilation_powers": list(self.backbone.dilation_powers),
            "mlp_hidden": hidden_dims,
        }


class TeacherPPGNetwork(BPRegressor):
    """PPG teacher N_PPG. Input shape: [B, 1, L]."""

    def __init__(self, **kwargs) -> None:
        super().__init__(input_rows=1, **kwargs)


class StudentRPPGNetwork(BPRegressor):
    """rPPG student N_rPPG. Input shape: [B, K, L]."""

    def __init__(self, k_signals: int = 15, **kwargs) -> None:
        super().__init__(input_rows=k_signals, **kwargs)
