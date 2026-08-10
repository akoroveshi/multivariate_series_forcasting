"""Model registry for the forecasting project.

* ``"lstm_attention"`` -- recurrent-attention hybrid, iterative block decoding
* ``"patchtst"``       -- patch Transformer, direct multi-step decoding
* ``"ensemble"``       -- weighted blend of trained members

Every model exposes the same two entry points::

    forward(history_target, history_features, future_features, static=None,
            future_target=None, teacher_forcing_ratio=0.0) -> (B, block_len, 1)
    rollout(history_target, history_features, future_features, static=None,
            horizon=None)                                  -> (B, horizon, 1)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from .ensemble import EnsembleForecaster
from .lstm_attention import LSTMAttentionForecaster
from .patchtst import PatchTST

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm_attention": LSTMAttentionForecaster,
    "patchtst": PatchTST,
    "ensemble": EnsembleForecaster,
}


def build_model(model_type: str, **kwargs: Any) -> nn.Module:
    """Instantiate a registered model, ignoring configuration keys it does not use."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type {model_type!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](**kwargs)


__all__ = [
    "EnsembleForecaster",
    "LSTMAttentionForecaster",
    "PatchTST",
    "MODEL_REGISTRY",
    "build_model",
]
