"""Rolling forecast generation and scoring.

The public release ships no validation labels, so every number reported in the
report is produced here on a held-out tail of ``train.csv``. Two regimes are
evaluated (see :func:`~src.features.split_positions`):

``val``
    The forecast window starts immediately after the last observed target -- the
    situation of the public validation leaderboard.

``test``
    The forecast window starts 336 steps after the last observed target, so the
    model has to bridge an unobserved gap first. This mirrors the private test
    split, where the released history ends before the validation window.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .dataset import build_inference_batch
from .features import SeriesPanel
from .metrics import compute_metrics


def _position_lookup(timestamps: np.ndarray) -> dict[np.datetime64, int]:
    return {stamp: position for position, stamp in enumerate(timestamps)}


@torch.no_grad()
def forecast_positions(
    model: torch.nn.Module,
    panel: SeriesPanel,
    targets: dict[object, tuple[int, Sequence[int]]],
    device: torch.device | str = "cpu",
    history_len: int | None = None,
    batch_size: int = 32,
    clip_min: float | None = None,
) -> dict[object, np.ndarray]:
    """Forecast the requested positions of each series.

    ``targets`` maps ``series_id -> (history_end, positions)`` where ``positions``
    are absolute indices into the series' time grid, all at or after
    ``history_end``. Series that need the same rollout length are batched
    together, which makes a full 96-series 336-step evaluation a handful of
    forward passes.
    """
    model.eval()
    history_len = history_len or int(getattr(model, "history_len", 168))
    grouped: dict[tuple[int, int], list[object]] = {}
    for series_id, (history_end, positions) in targets.items():
        horizon = int(max(positions)) - history_end + 1
        grouped.setdefault((history_end, horizon), []).append(series_id)

    predictions: dict[object, np.ndarray] = {}
    for (history_end, horizon), series_ids in grouped.items():
        for offset in range(0, len(series_ids), batch_size):
            chunk = series_ids[offset : offset + batch_size]
            batch = build_inference_batch(
                panel, chunk, history_end, history_len, horizon, device=device
            )
            rolled = model.rollout(
                batch["history_target"],
                batch["history_features"],
                batch["future_features"],
                static=batch["static"],
                horizon=horizon,
                future_history_features=batch["future_history_features"],
                series_index=batch["series_index"],
            )
            rolled = rolled.squeeze(-1).cpu().numpy()
            if clip_min is not None:
                rolled = np.maximum(rolled, clip_min)
            for row, series_id in enumerate(chunk):
                positions = np.asarray(targets[series_id][1], dtype=int) - history_end
                predictions[series_id] = rolled[row, positions]
    return predictions


def forecast_frame(
    model: torch.nn.Module,
    panel: SeriesPanel,
    forecast_index: pd.DataFrame,
    device: torch.device | str = "cpu",
    history_len: int | None = None,
    batch_size: int = 32,
    clip_min: float | None = None,
    history_end: dict[object, int] | None = None,
) -> pd.DataFrame:
    """Return a ``series_id,timestamp,prediction`` frame for a forecast index."""
    spec = panel.spec
    index = forecast_index.copy()
    index[spec.time_col] = pd.to_datetime(index[spec.time_col])

    targets: dict[object, tuple[int, list[int]]] = {}
    row_order: dict[object, pd.DataFrame] = {}
    for series_id, part in index.groupby(spec.series_col, sort=False):
        if series_id not in panel.series:
            raise KeyError(f"Series {series_id!r} is missing from the covariate panel.")
        arrays = panel[series_id]
        lookup = _position_lookup(arrays.timestamps)
        part = part.sort_values(spec.time_col)
        missing = [ts for ts in part[spec.time_col] if np.datetime64(ts) not in lookup]
        if missing:
            raise KeyError(
                f"{len(missing)} requested timestamps for series {series_id!r} are absent from the "
                "covariate table (first: %s)." % missing[0]
            )
        positions = [lookup[np.datetime64(ts)] for ts in part[spec.time_col]]
        start = (
            arrays.observed_length
            if history_end is None
            else int(history_end.get(series_id, arrays.observed_length))
        )
        start = min(start, int(min(positions)))
        if start <= 0:
            raise ValueError(
                f"Series {series_id!r} has no observed target before its first requested "
                "timestamp, so there is nothing to condition on. When running "
                "predict.py, use a checkpoint built by scripts/make_submission.py: it "
                "bundles a history tail with targets, which a raw training checkpoint "
                "does not carry."
            )
        targets[series_id] = (start, positions)
        row_order[series_id] = part

    predicted = forecast_positions(
        model,
        panel,
        {k: (v[0], v[1]) for k, v in targets.items()},
        device=device,
        history_len=history_len,
        batch_size=batch_size,
        clip_min=clip_min,
    )

    frames = []
    for series_id, part in row_order.items():
        out = part[[spec.series_col, spec.time_col]].copy()
        out["prediction"] = predicted[series_id]
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def score_frame(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "target",
) -> dict[str, float]:
    """Join predictions onto labels and compute all leaderboard metrics."""
    left = labels[[series_col, time_col, target_col]].copy()
    left[time_col] = pd.to_datetime(left[time_col])
    right = predictions[[series_col, time_col, "prediction"]].copy()
    right[time_col] = pd.to_datetime(right[time_col])
    merged = left.merge(right, on=[series_col, time_col], how="left", validate="one_to_one")
    if merged["prediction"].isna().any():
        raise ValueError(f"{int(merged['prediction'].isna().sum())} label rows have no prediction.")
    return compute_metrics(merged[target_col].to_numpy(), merged["prediction"].to_numpy())


def labels_from_panel(panel: SeriesPanel, positions: slice) -> pd.DataFrame:
    """Extract ``series_id,timestamp,target`` rows for a positional slice."""
    spec = panel.spec
    frames = []
    for series_id, arrays in panel.items():
        frames.append(
            pd.DataFrame(
                {
                    spec.series_col: series_id,
                    spec.time_col: arrays.timestamps[positions],
                    spec.target_col: arrays.y[positions],
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    return frame.loc[np.isfinite(frame[spec.target_col])].reset_index(drop=True)


def evaluate_split(
    model: torch.nn.Module,
    panel: SeriesPanel,
    positions: slice,
    history_end: int,
    device: torch.device | str = "cpu",
    history_len: int | None = None,
    batch_size: int = 32,
    clip_min: float | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Score a positional slice, conditioning on history up to ``history_end``."""
    labels = labels_from_panel(panel, positions)
    index = labels[[panel.spec.series_col, panel.spec.time_col]]
    ends = {series_id: history_end for series_id in panel.keys()}
    predictions = forecast_frame(
        model,
        panel,
        index,
        device=device,
        history_len=history_len,
        batch_size=batch_size,
        clip_min=clip_min,
        history_end=ends,
    )
    return score_frame(predictions, labels, *_columns(panel)), predictions


def _columns(panel: SeriesPanel) -> tuple[str, str, str]:
    spec = panel.spec
    return spec.series_col, spec.time_col, spec.target_col


@torch.no_grad()
def evaluate_windows(
    model: torch.nn.Module,
    panel: SeriesPanel,
    positions: slice,
    history_len: int,
    horizon: int,
    stride: int | None = None,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    clip_min: float | None = None,
) -> dict[str, float]:
    """Rolling-origin evaluation: many anchors, each with its own history window.

    This is the usual long-term-forecasting protocol and the one used for the
    additional dataset, whose held-out slices span thousands of steps: instead of
    a single enormous rollout, the forecast origin slides across the slice in
    ``stride`` increments and every origin is scored over ``horizon`` steps.
    """
    model.eval()
    stride = horizon if stride is None else stride
    series_ids = list(panel.keys())
    anchors = range(positions.start, positions.stop - horizon + 1, stride)
    truths: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    for anchor in anchors:
        for offset in range(0, len(series_ids), batch_size):
            chunk = series_ids[offset : offset + batch_size]
            usable = [
                sid
                for sid in chunk
                if anchor > 0 and np.isfinite(panel[sid].y[anchor : anchor + horizon]).all()
            ]
            if not usable:
                continue
            batch = build_inference_batch(panel, usable, anchor, history_len, horizon, device=device)
            rolled = model.rollout(
                batch["history_target"],
                batch["history_features"],
                batch["future_features"],
                static=batch["static"],
                horizon=horizon,
                future_history_features=batch["future_history_features"],
                series_index=batch["series_index"],
            )
            rolled = rolled.squeeze(-1).cpu().numpy()
            if clip_min is not None:
                rolled = np.maximum(rolled, clip_min)
            preds.append(rolled)
            truths.append(np.stack([panel[sid].y[anchor : anchor + horizon] for sid in usable]))
    if not preds:
        return {}
    return compute_metrics(np.concatenate(truths), np.concatenate(preds))
