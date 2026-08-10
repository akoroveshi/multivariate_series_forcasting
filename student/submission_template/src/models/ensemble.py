"""Weighted blending ensemble of independently trained forecasters.

The members are trained separately and then combined at *rollout* level: each
member covers the requested horizon with its own block length and decoding
strategy (iterative for the LSTM, direct for PatchTST) and the horizon-aligned
outputs are blended with fixed weights. Blending after the rollout rather than
per block matters, because the two families make structurally different errors:
the iterative decoder drifts smoothly while the direct model is noisier but
unbiased over long horizons.

Member weights are stored as a buffer, so they survive checkpoint round-trips
and the whole ensemble is a single ``checkpoint.pt``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn


class EnsembleForecaster(nn.Module):
    """Convex combination of several forecasters."""

    def __init__(
        self,
        members: Sequence[Mapping[str, Any]],
        member_weights: Sequence[float] | None = None,
        **_: object,
    ) -> None:
        super().__init__()
        from . import build_model  # local import: the registry lives in __init__

        if not members:
            raise ValueError("An ensemble needs at least one member configuration.")
        self.member_configs = [dict(cfg) for cfg in members]
        self.members = nn.ModuleList(
            [
                build_model(cfg["model_type"], **{k: v for k, v in cfg.items() if k != "model_type"})
                for cfg in self.member_configs
            ]
        )
        weights = (
            torch.ones(len(self.members))
            if member_weights is None
            else torch.tensor(list(member_weights), dtype=torch.float32)
        )
        if weights.numel() != len(self.members):
            raise ValueError("member_weights length must match the number of members.")
        self.register_buffer("weights", weights / weights.sum())
        self.block_len = min(int(getattr(m, "block_len", 24)) for m in self.members)

    def set_weights(self, member_weights: Sequence[float]) -> None:
        """Replace the blending weights (renormalised to sum to one)."""
        weights = torch.tensor(list(member_weights), dtype=torch.float32)
        if weights.numel() != len(self.members):
            raise ValueError("member_weights length must match the number of members.")
        self.weights.copy_(weights / weights.sum())

    def _blend(self, outputs: Sequence[tuple[int, torch.Tensor]]) -> torch.Tensor:
        horizon = min(out.size(1) for _, out in outputs)
        stacked = torch.stack([out[:, :horizon, :] for _, out in outputs], dim=0)
        weights = self.weights[[index for index, _ in outputs]].view(-1, 1, 1, 1)
        return (stacked * weights.to(stacked.dtype)).sum(dim=0)

    def _active(self, tolerance: float = 1e-8) -> list[int]:
        """Indices of members with non-negligible weight.

        A grid search can legitimately return a degenerate blend (all the weight on
        one member). Skipping the others then saves their entire rollout, which for
        the iterative member is most of the inference cost.
        """
        active = [i for i, w in enumerate(self.weights.tolist()) if abs(w) > tolerance]
        return active or [int(self.weights.argmax())]

    def forward(
        self,
        history_target: torch.Tensor,
        history_features: torch.Tensor,
        future_features: torch.Tensor,
        static: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Blend member rollouts over the requested horizon."""
        return self.rollout(
            history_target,
            history_features,
            future_features,
            static=static,
            horizon=kwargs.get("horizon", future_features.size(1)),
            future_history_features=kwargs.get("future_history_features"),
            series_index=kwargs.get("series_index"),
        )

    @torch.no_grad()
    def rollout(
        self,
        history_target: torch.Tensor,
        history_features: torch.Tensor,
        future_features: torch.Tensor,
        static: torch.Tensor | None = None,
        horizon: int | None = None,
        future_history_features: torch.Tensor | None = None,
        series_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        horizon = future_features.size(1) if horizon is None else horizon
        outputs = [
            (
                index,
                self.members[index].rollout(
                    history_target,
                    history_features,
                    future_features,
                    static=static,
                    horizon=horizon,
                    future_history_features=future_history_features,
                    series_index=series_index,
                ),
            )
            for index in self._active()
        ]
        return self._blend(outputs)
