"""Leaderboard metrics.

Mirrors ``evaluate_predictions.py`` of the course Hugging Face Space so that
locally reported numbers are directly comparable to leaderboard scores. WAPE is
the primary metric::

    WAPE = 100 * sum |y - y_hat| / sum |y|

Because the denominator is a single global constant, WAPE is a *rescaled sum of
absolute errors*. Minimising plain L1 loss in the original target units is
therefore the loss-metric-aligned training objective -- in particular the loss
must not be computed on per-window normalised targets, which would implicitly
reweight series by their own scale.
"""

from __future__ import annotations

import numpy as np
import torch

EPSILON = 1e-8
SUPPORTED_METRICS = ("mae", "mse", "rmse", "mape", "smape", "wape")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return every leaderboard metric for a pair of aligned arrays."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}.")
    error = np.abs(y_true - y_pred)
    return {
        "mae": float(np.mean(error)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(np.mean(error / np.maximum(np.abs(y_true), EPSILON)) * 100.0),
        "smape": float(
            np.mean(2.0 * error / np.maximum(np.abs(y_true) + np.abs(y_pred), EPSILON)) * 100.0
        ),
        "wape": float(np.sum(error) / max(float(np.sum(np.abs(y_true))), EPSILON) * 100.0),
    }


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error in percent (primary metric)."""
    return compute_metrics(y_true, y_pred)["wape"]


def format_metrics(metrics: dict[str, float]) -> str:
    """Compact one-line rendering used by the training logs."""
    return "  ".join(f"{name}={metrics[name]:.4f}" for name in SUPPORTED_METRICS if name in metrics)


class WapeLoss(torch.nn.Module):
    """Differentiable batch-level WAPE, i.e. L1 normalised by the batch target mass.

    Equivalent to L1 up to a per-batch constant when target magnitudes are
    homogeneous, but slightly more stable when a batch mixes large and small
    series because the gradient scale no longer depends on the batch composition.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (prediction - target).abs().sum() / (target.abs().sum() + self.eps)


def build_loss(name: str) -> torch.nn.Module:
    """Resolve a training loss by name."""
    losses = {
        "l1": torch.nn.L1Loss(),
        "mae": torch.nn.L1Loss(),
        "mse": torch.nn.MSELoss(),
        "huber": torch.nn.HuberLoss(delta=1.0),
        "wape": WapeLoss(),
    }
    if name not in losses:
        raise ValueError(f"Unknown loss {name!r}. Available: {sorted(losses)}")
    return losses[name]
