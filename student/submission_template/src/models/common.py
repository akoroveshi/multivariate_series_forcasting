"""Shared building blocks used by the forecasting models.
- RevIN             -> Reversible Instance Normalization (Kim et al., 2022)
- SpatialDropout1d  -> channel-wise dropout (Tompson et al., 2015)
- calendar_features -> cyclical hour-of-day / day-of-week encodings
"""

from __future__ import annotations
import math
import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible Instance Normalization.
    Normalizes each series independently over the time dimension using its
    own mean/std
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Store instance statistics from ``x`` and return the normalized tensor """
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        self._mean, self._std = mean, std
        out = (x - mean) / std
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert ``normalize`` using the statistics captured on the last call """
        if self._mean is None or self._std is None:
            raise RuntimeError("normalize() must be called before denormalize().")
        out = x
        if self.affine:
            out = (out - self.bias) / (self.weight + self.eps)
        return out * self._std + self._mean


class SpatialDropout1d(nn.Module):
    """Drops entire feature channels rather than individual timesteps """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1).permute(0, 2, 1, 3)
        x = self.dropout(x)
        return x.permute(0, 2, 1, 3).squeeze(-1)


def calendar_features(timestamps: torch.Tensor) -> torch.Tensor:
    """Build cyclical hour-of-day / day-of-week features from epoch-hour indices """
    t = timestamps.float()
    hour = t % 24
    dow = (t // 24) % 7
    hour_sin = torch.sin(2 * math.pi * hour / 24)
    hour_cos = torch.cos(2 * math.pi * hour / 24)
    dow_sin = torch.sin(2 * math.pi * dow / 7)
    dow_cos = torch.cos(2 * math.pi * dow / 7)
    return torch.stack([hour_sin, hour_cos, dow_sin, dow_cos], dim=-1)