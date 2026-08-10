"""Checkpoint-level wrapper around the registered forecasters.

``ForecastModel`` bundles everything ``predict.py`` needs to reproduce training
time behaviour from a single file:

* the network weights and the configuration required to rebuild the architecture,
* the :class:`~src.features.FeatureSpec` and :class:`~src.features.FeatureScaler`
  (covariate standardisation statistics fitted on the training slice),
* optionally a *context table*: the tail of the public history plus the public
  validation covariates.

The context table matters because of the shape of the private evaluation. The
private input directory contains ``test_input.csv`` (covariates for the test
window) and ``forecast_index_test.csv``, but no observed target: the last
released target sits 336 steps before the first test timestamp. Bundling the
public history tail lets ``predict.py`` reconstruct a continuous covariate
timeline and roll the model forward across that gap without touching the network
or the internet.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from .features import FeatureScaler, FeatureSpec
from .models import build_model

DEFAULT_CONFIG: dict[str, Any] = {
    "model_type": "lstm_attention",
    "num_history_features": 0,
    "num_future_features": None,
    "num_static_features": 0,
    "hidden_size": 128,
    "encoder_layers": 2,
    "decoder_layers": 1,
    "dropout": 0.1,
    "block_len": 24,
    "history_len": 168,
}


class ForecastModel(torch.nn.Module):
    """Wraps the active forecaster and exposes a stable inference interface."""

    def __init__(self, **config: Any) -> None:
        super().__init__()
        merged = {**DEFAULT_CONFIG, **config}
        if merged["model_type"] == "ensemble":
            # Ensembles carry their own member configs; drop the LSTM defaults.
            merged = {
                k: v
                for k, v in merged.items()
                if k
                in {
                    "model_type",
                    "members",
                    "member_weights",
                    "history_len",
                    "block_len",
                    "clip_min",
                }
            }
        self.config = merged
        self.feature_spec: FeatureSpec | None = None
        self.feature_scaler: FeatureScaler | None = None
        self.series_index: dict[str, int] | None = None
        self.context: pd.DataFrame | None = None
        self.net = build_model(
            self.config["model_type"],
            **{k: v for k, v in self.config.items() if k != "model_type"},
        )

    # ------------------------------------------------------------- properties
    @property
    def block_len(self) -> int:
        """Steps produced by one forward pass of the underlying network."""
        return int(getattr(self.net, "block_len", self.config.get("block_len", 24)))

    @property
    def history_len(self) -> int:
        """Conditioning window length the model was trained with."""
        return int(self.config.get("history_len", getattr(self.net, "history_len", 168)))

    # ---------------------------------------------------------------- forward
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.net(*args, **kwargs)

    def rollout(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.net.rollout(*args, **kwargs)

    # ------------------------------------------------------------ checkpoints
    def attach_preprocessing(self, spec: FeatureSpec, scaler: FeatureScaler) -> None:
        """Record the preprocessing used during training."""
        self.feature_spec = spec
        self.feature_scaler = scaler

    def attach_series_index(self, series_index: dict[Any, int] | None) -> None:
        """Record the ``series_id -> embedding row`` mapping used during training."""
        self.series_index = (
            {str(key): int(value) for key, value in series_index.items()}
            if series_index is not None
            else None
        )

    def attach_context(self, context: pd.DataFrame | None) -> None:
        """Record the history/covariate tail bundled into the checkpoint."""
        self.context = context

    def save_checkpoint(self, path: str) -> None:
        payload: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "config": self.config,
        }
        if self.feature_spec is not None:
            payload["feature_spec"] = self.feature_spec.to_dict()
        if self.feature_scaler is not None:
            payload["feature_scaler"] = self.feature_scaler.to_dict()
        if self.series_index is not None:
            payload["series_index"] = self.series_index
        if self.context is not None:
            payload["context"] = {
                "columns": list(self.context.columns),
                "records": {
                    col: self.context[col].to_numpy() for col in self.context.columns
                },
            }
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = "cpu") -> "ForecastModel":
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise ValueError("Checkpoint must be a dict containing 'state_dict' and 'config'.")
        model = cls(**checkpoint.get("config", {}))
        model.load_state_dict(checkpoint["state_dict"])
        if "feature_spec" in checkpoint:
            model.feature_spec = FeatureSpec.from_dict(checkpoint["feature_spec"])
        if "feature_scaler" in checkpoint:
            model.feature_scaler = FeatureScaler.from_dict(checkpoint["feature_scaler"])
        if "series_index" in checkpoint:
            model.series_index = {
                str(key): int(value) for key, value in checkpoint["series_index"].items()
            }
        if "context" in checkpoint:
            payload = checkpoint["context"]
            model.context = pd.DataFrame(
                {col: payload["records"][col] for col in payload["columns"]}
            )
        model.eval()
        return model


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters, reported in the experiments table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
