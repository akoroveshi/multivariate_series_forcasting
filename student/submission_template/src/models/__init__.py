"""Model registry for the forecasting project.
- "lstm_attention": recurrent-attention hybrid 
- "patchtst": patch-based Transformer 
"""

from __future__ import annotations
from typing import Any
from torch import nn

from .lstm_attention import LSTMAttentionForecaster
from .patchtst import PatchTST

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm_attention": LSTMAttentionForecaster,
    "patchtst": PatchTST,
}


def build_model(model_type: str, **kwargs: Any) -> nn.Module:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)


__all__ = ["LSTMAttentionForecaster", "PatchTST", "MODEL_REGISTRY", "build_model"]