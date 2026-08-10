"""Shared building blocks used by the forecasting models.

* :class:`RevIN`             -- Reversible Instance Normalization (Kim et al., 2022)
* :class:`SpatialDropout1d`  -- channel-wise dropout (Tompson et al., 2015)
* :func:`chained_rollout`    -- horizon extension by chaining fixed-size blocks

``RevIN`` is written *statelessly*: :meth:`RevIN.stats` returns the instance
statistics and both :meth:`normalize` and :meth:`denormalize` take them as an
argument. Caching the statistics on the module (as is common in reference
implementations) silently breaks autoregressive decoders, because normalizing
the single previous step overwrites the statistics of the full history window
with ``mean = value, std = 0``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

import torch
from torch import nn


@contextmanager
def inference_mode(module: nn.Module) -> Iterator[None]:
    """Temporarily switch ``module`` to eval mode, restoring the previous state.

    A rollout must be deterministic: dropout left active would make each chained
    block sample a different sub-network, so two calls with identical inputs would
    disagree. Guarding inside ``rollout`` means callers cannot forget ``eval()``.
    """
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class RevIN(nn.Module):
    """Reversible instance normalization over the time axis.

    Each window is normalized by its own mean/standard deviation, which removes
    the dominant part of the distribution shift between training windows and the
    forecast window, and the transformation is inverted on the model output.
    """

    def __init__(self, num_features: int = 1, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` of ``x`` over the time axis, shape ``(B, 1, C)``."""
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        return mean, std

    def normalize(self, x: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, std = stats
        out = (x - mean) / std
        if self.affine:
            out = out * self.weight + self.bias
        return out

    def denormalize(self, x: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, std = stats
        out = x
        if self.affine:
            out = (out - self.bias) / (self.weight + self.eps)
        return out * std + mean


class SpatialDropout1d(nn.Module):
    """Drop whole feature channels of a ``(B, T, C)`` sequence.

    Standard dropout removes individual ``(t, c)`` entries, which a recurrent
    encoder can trivially reconstruct from neighbouring timesteps. Dropping a
    channel for the entire window instead forces the model to spread its
    reliance across covariates.
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout1d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        return self.dropout(x.transpose(1, 2)).transpose(1, 2)


class SeriesEmbedding(nn.Module):
    """Learned per-series vector concatenated to the static covariates.

    The benchmark's static columns (capacity, zone) explain very little: a
    per-series ordinary least squares fit on the covariates is markedly better
    than a pooled one, i.e. the covariate-to-load mapping differs per unit. A
    free embedding gives the shared backbone that unit-specific degree of freedom
    while keeping one model for all 96 series. Row ``0`` is reserved for series
    that were not seen during training, so the checkpoint degrades gracefully
    instead of raising if the private split introduces a new unit.
    """

    def __init__(self, num_slots: int, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.table = nn.Embedding(max(num_slots, 1), dim)
        nn.init.normal_(self.table.weight, std=0.02)
        with torch.no_grad():
            self.table.weight[0].zero_()

    def forward(self, series_index: torch.Tensor | None, batch_size: int, device) -> torch.Tensor:
        if series_index is None:
            return torch.zeros(batch_size, self.dim, device=device)
        clamped = series_index.clamp(min=0, max=self.table.num_embeddings - 1)
        return self.table(clamped)


@torch.no_grad()
def chained_rollout(
    predict_block: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    history_target: torch.Tensor,
    history_features: torch.Tensor,
    future_features: torch.Tensor,
    horizon: int,
    block_len: int,
    history_len: int | None = None,
    future_history_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cover ``horizon`` steps by chaining ``block_len``-step predictions.

    After every block the predicted values are appended to the conditioning
    window -- together with encoder-shaped covariate rows for those steps, taken
    from ``future_history_features`` (defaults to ``future_features`` when the
    known-future and encoder feature sets coincide, as they do on this
    benchmark). The window is then trimmed back to ``history_len``. Blocks
    shorter than ``block_len`` are right-padded with the last known covariate row
    and truncated afterwards, so the model always sees the shape it was trained
    on.
    """
    if history_len is None:
        history_len = history_target.size(1)
    if future_history_features is None:
        if future_features.size(-1) != history_features.size(-1):
            raise ValueError(
                "future_history_features is required when the encoder and decoder "
                "feature sets differ."
            )
        future_history_features = future_features
    target = history_target
    features = history_features
    outputs: list[torch.Tensor] = []
    cursor = 0
    while cursor < horizon:
        step = min(block_len, horizon - cursor)
        block_features = future_features[:, cursor : cursor + block_len, :]
        if block_features.size(1) < block_len:
            pad = block_len - block_features.size(1)
            block_features = torch.cat(
                [block_features, block_features[:, -1:, :].expand(-1, pad, -1)], dim=1
            )
        prediction = predict_block(
            target[:, -history_len:, :], features[:, -history_len:, :], block_features
        )
        prediction = prediction[:, :step, :]
        outputs.append(prediction)
        target = torch.cat([target, prediction], dim=1)
        features = torch.cat(
            [features, future_history_features[:, cursor : cursor + step, :]], dim=1
        )
        cursor += step
    return torch.cat(outputs, dim=1)
