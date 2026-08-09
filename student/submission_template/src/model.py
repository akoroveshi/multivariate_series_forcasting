"""Forecasting model used by predict.py and train.py"""

from __future__ import annotations
from typing import Any
import torch
from torch import nn

from .models import build_model 

DEFAULT_CONFIG: dict[str, Any] = {
    "model_type": "lstm_attention",
    "num_calendar_features": 4,
    "hidden_size": 128,
    "encoder_layers": 2,
    "decoder_layers": 1,
    "dropout": 0.1,
    "block_len": 24
}

class ForecastModel(torch.nn.Module):
    """Wraps active forecaster and exposes a stable interface for inference"""

    def __init__(self, **config: Any) -> None:
        """Create the placeholder one-parameter model."""
        super().__init__()
        self.config = {**DEFAULT_CONFIG, **config}
        model_kwargs = {k: v for k, v in self.config.items() if k != "model_type"}
        self.net = build_model(self.config["model_type"], **model_kwargs)
    
    @property
    def block_len(self) -> int:
        """ Number of steps produced per decoder rollout block """
        return self.config["block_len"]

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.net(*args, **kwargs)
    
    def rollout(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.net.rollout(*args, **kwargs) 
    
    def save_checkpoint(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.config}, path)
    
    @classmethod 
    def load_checkpoint(cls, path: str, map_location: str = "cpu") -> "ForecastModel":
        checkpoint = torch.load(path, map_location = map_location, weights_only = False)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            config = checkpoint.get("config", {})
            model = cls(**config)
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict):
            model = cls()
            model.load_state_dict(checkpoint)
        else:
            raise ValueError("Checkpoint should be a state_dict")
        model.eval()
        return model