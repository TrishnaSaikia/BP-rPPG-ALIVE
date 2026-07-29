"""Paper-aligned modified temporal convolutional network (M-TCN).

The accepted paper describes the following operations:

* input temporal map of size K x L;
* one dilation block for dilation 1, followed by blocks with dilations
  2, 4, ..., 2^floor(log2(L));
* three causal convolutions per block;
* five 1 x 3 filters inside a block;
* chomp, ReLU, dropout, a residual/skip path, and a downsampling layer;
* final K x L map projected by a dense layer with L neurons to a 1 x L
  BP-relevant feature representation.

The paper does not state the exact operator used by the block's downsampling
layer. Here it is implemented as a 1 x 1 channel projection from the five
internal feature maps back to one map. This preserves the K x L dimensions
reported between successive dilation blocks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


class Chomp2d(nn.Module):
    """Remove right-side temporal padding introduced by causal Conv2d."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        if chomp_size < 0:
            raise ValueError("chomp_size must be non-negative")
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()


class PaperDilationBlock(nn.Module):
    """One paper-described dilation block.

    Input/output shape: [B, 1, K, L]. Internally, five feature maps are
    produced by three causal 1 x 3 convolutions. A skip projection is added,
    followed by a 1 x 1 projection back to one K x L map.
    """

    def __init__(
        self,
        internal_filters: int = 5,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.01,
    ) -> None:
        super().__init__()
        if internal_filters <= 0:
            raise ValueError("internal_filters must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        temporal_padding = (kernel_size - 1) * dilation

        def causal_conv(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, kernel_size),
                    dilation=(1, dilation),
                    padding=(0, temporal_padding),
                ),
                Chomp2d(temporal_padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.conv1 = causal_conv(1, internal_filters)
        self.conv2 = causal_conv(internal_filters, internal_filters)
        self.conv3 = causal_conv(internal_filters, internal_filters)
        self.skip = nn.Conv2d(1, internal_filters, kernel_size=1)
        self.residual_activation = nn.ReLU()
        self.downsample = nn.Conv2d(internal_filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.residual_activation(out + residual)
        return self.downsample(out)


class PaperMTCN(nn.Module):
    """M-TCN feature extractor that returns the paper's 1 x L feature map."""

    def __init__(
        self,
        input_rows: int,
        clip_samples: int,
        internal_filters: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.01,
        dilation_powers: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if input_rows <= 0:
            raise ValueError("input_rows must be positive")
        if clip_samples <= 1:
            raise ValueError("clip_samples must be greater than one")

        self.input_rows = input_rows
        self.clip_samples = clip_samples

        if dilation_powers is None:
            max_power = int(math.floor(math.log2(clip_samples)))
            dilation_powers = list(range(max_power + 1))
        if not dilation_powers:
            raise ValueError("At least one dilation block is required")

        self.dilation_powers = tuple(int(power) for power in dilation_powers)
        self.blocks = nn.Sequential(
            *[
                PaperDilationBlock(
                    internal_filters=internal_filters,
                    kernel_size=kernel_size,
                    dilation=2**power,
                    dropout=dropout,
                )
                for power in self.dilation_powers
            ]
        )

        # K x L -> 1 x L, as described in the manuscript.
        self.feature_projection = nn.Linear(input_rows * clip_samples, clip_samples)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, K, L], received {tuple(x.shape)}")
        if x.shape[1] != self.input_rows or x.shape[2] != self.clip_samples:
            raise ValueError(
                "Input shape mismatch: expected "
                f"[B, {self.input_rows}, {self.clip_samples}], received {tuple(x.shape)}"
            )
        out = self.blocks(x.unsqueeze(1))
        out = out.flatten(start_dim=1)
        return self.feature_projection(out)
