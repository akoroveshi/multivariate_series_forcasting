"""Multivariate time series forecasting package (DL bonus project, group 140).

Modules
-------
``features``  covariate groups, missing-value handling, standardisation, panels
``dataset``   sliding-window dataset and inference batching
``models``    LSTM+Attention hybrid, PatchTST, blending ensemble
``model``     ``ForecastModel`` checkpoint wrapper used by ``predict.py``
``metrics``   leaderboard metrics (WAPE primary) and training losses
``train``     training / model-selection entrypoint
``evaluate``  rolling forecast generation and scoring
``baselines`` naive, lag and seasonal reference forecasts
"""

from __future__ import annotations

__all__ = [
    "baselines",
    "dataset",
    "evaluate",
    "features",
    "metrics",
    "model",
    "models",
    "train",
]
