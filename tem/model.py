from __future__ import annotations

import torch
from torch import nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class FastRULNet(nn.Module):
    """Small high-throughput Conv1D model for RUL regression."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        depth: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[DepthwiseSeparableBlock(hidden, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

