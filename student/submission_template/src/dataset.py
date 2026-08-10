"""Sliding-window dataset construction.

A training sample is a ``(history, future)`` pair drawn from one series:

* ``history_target``   -- ``(L, 1)`` observed target over the conditioning window
* ``history_features`` -- ``(L, F_h)`` all covariates over the same window
* ``future_features``  -- ``(H, F_f)`` covariates over the forecast window that
  are *known in advance*. For this benchmark every covariate is known, so
  ``F_f == F_h``; for the additional dataset only the calendar encodings are,
  and ``F_f < F_h``.
* ``static``           -- ``(S,)`` per-series constants
* ``future_target``    -- ``(H, 1)`` supervision signal

``gap`` inserts unobserved steps between the two windows. The private test split
starts 336 steps after the last released target, so evaluating with ``gap > 0``
measures exactly that regime.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import SeriesPanel


class WindowDataset(Dataset):
    """Windows over a :class:`~src.features.SeriesPanel`."""

    def __init__(
        self,
        panel: SeriesPanel,
        history_len: int = 168,
        horizon: int = 336,
        stride: int = 1,
        gap: int = 0,
        start_position: int = 0,
        end_position: int | None = None,
    ) -> None:
        self.panel = panel
        self.history_len = history_len
        self.horizon = horizon
        self.gap = gap
        self.future_index = panel.future_index
        self._index: list[tuple[object, int]] = []

        span = history_len + gap + horizon
        for series_id, arrays in panel.items():
            limit = len(arrays) if end_position is None else min(len(arrays), end_position)
            finite = np.isfinite(arrays.y)
            for start in range(start_position, limit - span + 1, stride):
                hist = slice(start, start + history_len)
                fut_start = start + history_len + gap
                fut = slice(fut_start, fut_start + horizon)
                if finite[hist].all() and finite[fut].all():
                    self._index.append((series_id, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        series_id, start = self._index[idx]
        arrays = self.panel[series_id]
        hist = slice(start, start + self.history_len)
        fut_start = start + self.history_len + self.gap
        fut = slice(fut_start, fut_start + self.horizon)
        as_tensor = lambda values: torch.from_numpy(np.ascontiguousarray(values))  # noqa: E731
        return {
            "history_target": as_tensor(arrays.y[hist]).unsqueeze(-1),
            "history_features": as_tensor(arrays.x[hist]),
            "future_features": as_tensor(arrays.x[fut][:, self.future_index]),
            "static": as_tensor(arrays.static),
            "series_index": torch.tensor(self.panel.index_of(series_id), dtype=torch.long),
            "future_target": as_tensor(arrays.y[fut]).unsqueeze(-1),
        }


def build_inference_batch(
    panel: SeriesPanel,
    series_ids: Sequence[object],
    history_end: int,
    history_len: int,
    horizon: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Stack one inference window per series into a single batch.

    ``history_end`` is the exclusive end position of the conditioning window; the
    forecast covers positions ``[history_end, history_end + horizon)``. Series
    shorter than ``history_len`` are left-padded by repeating their first
    observation so the encoder always receives a full window.

    Besides ``future_features`` (the known-in-advance covariates) the batch also
    carries ``future_history_features``: encoder-shaped rows for the forecast
    steps, used when an iterative decoder appends a predicted block back onto its
    conditioning window. Past-only columns cannot be known there, so they are
    filled by persistence of the last observed value -- never by their true
    future values, which would leak the answer.
    """
    future_index = panel.future_index
    past_only_index = panel.past_only_index
    hist_y, hist_x, fut_x, fut_hx, static, index = [], [], [], [], [], []
    for series_id in series_ids:
        arrays = panel[series_id]
        start = max(0, history_end - history_len)
        y = arrays.y[start:history_end]
        x = arrays.x[start:history_end]
        if len(y) < history_len:
            pad = history_len - len(y)
            y = np.concatenate([np.repeat(y[:1], pad), y])
            x = np.concatenate([np.repeat(x[:1], pad, axis=0), x])

        future_rows = arrays.x[history_end : history_end + horizon]
        if len(future_rows) < horizon:
            pad = horizon - len(future_rows)
            future_rows = np.concatenate([future_rows, np.repeat(future_rows[-1:], pad, axis=0)])
        encoder_rows = future_rows.copy()
        if len(past_only_index):
            encoder_rows[:, past_only_index] = x[-1, past_only_index]

        hist_y.append(y)
        hist_x.append(x)
        fut_x.append(future_rows[:, future_index])
        fut_hx.append(encoder_rows)
        static.append(arrays.static)
        index.append(panel.index_of(series_id))

    to_tensor = lambda values: torch.from_numpy(np.stack(values).astype(np.float32)).to(device)  # noqa: E731
    return {
        "history_target": to_tensor(hist_y).unsqueeze(-1),
        "history_features": to_tensor(hist_x),
        "future_features": to_tensor(fut_x),
        "future_history_features": to_tensor(fut_hx),
        "static": to_tensor(static),
        "series_index": torch.tensor(index, dtype=torch.long, device=device),
    }
