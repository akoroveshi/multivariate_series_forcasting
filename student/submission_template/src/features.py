"""Feature pipeline for the operations forecasting benchmark.

The benchmark tables ship 22 covariates alongside the target. Inspecting
``validation_input.csv`` shows that it covers *exactly* the rows of
``forecast_index_validation.csv``: every covariate is therefore **known in
advance** for the forecast window, while the target is not. The models in
``src/models`` exploit that by conditioning the decoder on future covariates
instead of relying purely on an autoregressive rollout of the target.

The columns split into three groups:

* ``CALENDAR_FEATURES``  -- deterministic time encodings (also re-derivable
  from the timestamp, see :func:`derive_calendar_features`).
* ``KNOWN_DYNAMIC_FEATURES`` -- exogenous per-hour drivers. Ten of them are
  missing at random for roughly 4.5% of the rows, so each one gets an explicit
  missingness indicator on top of an interpolated value.
* ``STATIC_FEATURES`` -- constant per series (capacity, zone encoding).

Everything here is deliberately framework agnostic: it turns long CSV tables
into per-series float32 arrays that :mod:`src.dataset` slices into windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import math
import numpy as np
import pandas as pd

SERIES_COL = "series_id"
TIME_COL = "timestamp"
TARGET_COL = "target"
MASK_SUFFIX = "__isna"

CALENDAR_FEATURES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "trend",
)

KNOWN_DYNAMIC_FEATURES: tuple[str, ...] = (
    "workload_intensity",
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "promotion_intensity",
    "shock_risk",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
)

STATIC_FEATURES: tuple[str, ...] = ("nominal_capacity", "zone_sin", "zone_cos")

#: Covariates that contain missing values in the released benchmark tables.
DEFAULT_MASK_FEATURES: tuple[str, ...] = (
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "shock_risk",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
)


def derive_calendar_features(frame: pd.DataFrame, *, time_col: str = TIME_COL) -> pd.DataFrame:
    """Add cyclical calendar columns for tables that do not ship them.

    Used for the additional (Jena climate) dataset and as a safety net if a
    future release of the benchmark drops the pre-computed encodings.
    """
    result = frame.copy()
    stamps = pd.to_datetime(result[time_col])
    hour = stamps.dt.hour + stamps.dt.minute / 60.0
    dow = stamps.dt.dayofweek
    doy = stamps.dt.dayofyear
    result["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    result["dow_sin"] = np.sin(2 * math.pi * dow / 7.0)
    result["dow_cos"] = np.cos(2 * math.pi * dow / 7.0)
    result["doy_sin"] = np.sin(2 * math.pi * doy / 365.25)
    result["doy_cos"] = np.cos(2 * math.pi * doy / 365.25)
    result["is_weekend"] = (dow >= 5).astype("float32")
    # Rank *timestamps*, not rows: on a multi-series table every series shares the
    # same instants, so ranking rows would hand the 96 rows of one hour 96 different
    # trend values and turn a time feature into a series identifier.
    unique = np.unique(stamps.to_numpy())
    ramp = np.linspace(-1.0, 1.0, len(unique)) if len(unique) > 1 else np.zeros(len(unique))
    result["trend"] = ramp[np.searchsorted(unique, stamps.to_numpy())]
    return result


@dataclass(frozen=True)
class FeatureSpec:
    """Which columns act as covariates, and which of them are known in advance.

    ``dynamic`` covariates are available for the forecast window (all 22 columns
    of the benchmark), whereas ``past_only`` covariates are observed alongside
    the target and therefore unknown at prediction time. The distinction is what
    lets the same architecture serve both this benchmark and the classical
    long-term forecasting setting used for the additional dataset, where every
    channel except the calendar encodings is past-only.
    """

    dynamic: tuple[str, ...]
    past_only: tuple[str, ...] = ()
    static: tuple[str, ...] = ()
    mask_features: tuple[str, ...] = ()
    series_col: str = SERIES_COL
    time_col: str = TIME_COL
    target_col: str = TARGET_COL

    @classmethod
    def benchmark(cls) -> "FeatureSpec":
        """Feature layout of the course benchmark dataset (all covariates known)."""
        return cls(
            dynamic=CALENDAR_FEATURES + KNOWN_DYNAMIC_FEATURES,
            static=STATIC_FEATURES,
            mask_features=DEFAULT_MASK_FEATURES,
        )

    @property
    def covariates(self) -> tuple[str, ...]:
        """All raw covariate columns that must be present in the input table."""
        return tuple(self.dynamic) + tuple(self.past_only)

    @property
    def history_names(self) -> tuple[str, ...]:
        """Feature columns the encoder sees (in matrix order)."""
        return self.covariates + tuple(f"{c}{MASK_SUFFIX}" for c in self.mask_features)

    @property
    def future_names(self) -> tuple[str, ...]:
        """Feature columns available for the forecast window."""
        known = set(self.dynamic)
        return tuple(self.dynamic) + tuple(
            f"{c}{MASK_SUFFIX}" for c in self.mask_features if c in known
        )

    @property
    def future_index(self) -> tuple[int, ...]:
        """Positions of :attr:`future_names` inside :attr:`history_names`."""
        lookup = {name: i for i, name in enumerate(self.history_names)}
        return tuple(lookup[name] for name in self.future_names)

    @property
    def past_only_index(self) -> tuple[int, ...]:
        """Positions of history features that are unknown for the forecast window."""
        known = set(self.future_names)
        return tuple(i for i, name in enumerate(self.history_names) if name not in known)

    @property
    def num_history_features(self) -> int:
        return len(self.history_names)

    @property
    def num_future_features(self) -> int:
        return len(self.future_names)

    @property
    def num_static_features(self) -> int:
        return len(self.static)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dynamic": list(self.dynamic),
            "past_only": list(self.past_only),
            "static": list(self.static),
            "mask_features": list(self.mask_features),
            "series_col": self.series_col,
            "time_col": self.time_col,
            "target_col": self.target_col,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSpec":
        return cls(
            dynamic=tuple(payload["dynamic"]),
            past_only=tuple(payload.get("past_only", ())),
            static=tuple(payload.get("static", ())),
            mask_features=tuple(payload.get("mask_features", ())),
            series_col=payload.get("series_col", SERIES_COL),
            time_col=payload.get("time_col", TIME_COL),
            target_col=payload.get("target_col", TARGET_COL),
        )


@dataclass
class FeatureScaler:
    """Global mean/std standardisation for covariates.

    Statistics are fitted on the training slice only and then serialised into
    the checkpoint, so inference reproduces training-time preprocessing byte for
    byte without needing the training CSV.
    """

    means: dict[str, float]
    stds: dict[str, float]
    medians: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str]) -> "FeatureScaler":
        present = [c for c in columns if c in frame.columns]
        means, stds, medians = {}, {}, {}
        for col in present:
            values = pd.to_numeric(frame[col], errors="coerce")
            means[col] = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std(ddof=0)) if values.notna().any() else 1.0
            stds[col] = std if std > 1e-8 else 1.0
            medians[col] = float(values.median()) if values.notna().any() else 0.0
        return cls(means=means, stds=stds, medians=medians)

    def standardize(self, values: np.ndarray, column: str) -> np.ndarray:
        mean = self.means.get(column, 0.0)
        std = self.stds.get(column, 1.0)
        return (values - mean) / std

    def median(self, column: str) -> float:
        return self.medians.get(column, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"means": self.means, "stds": self.stds, "medians": self.medians}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureScaler":
        return cls(
            means=dict(payload["means"]),
            stds=dict(payload["stds"]),
            medians=dict(payload.get("medians", {})),
        )


@dataclass
class SeriesArrays:
    """Dense per-series arrays on a regular time grid."""

    series_id: Any
    timestamps: np.ndarray  # datetime64[ns], strictly increasing
    y: np.ndarray  # (T,) float32, NaN where the target is unknown
    x: np.ndarray  # (T, F) float32 dynamic covariates (standardised)
    static: np.ndarray  # (S,) float32 static covariates (standardised)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def observed_length(self) -> int:
        """Number of leading steps with an observed target."""
        finite = np.isfinite(self.y)
        if not finite.any():
            return 0
        return int(np.max(np.nonzero(finite)[0]) + 1)


class SeriesPanel:
    """Collection of :class:`SeriesArrays` sharing one feature layout."""

    def __init__(
        self,
        series: dict[Any, SeriesArrays],
        spec: FeatureSpec,
        scaler: FeatureScaler,
        series_index: Mapping[Any, int] | None = None,
    ) -> None:
        self.series = series
        self.spec = spec
        self.scaler = scaler
        #: ``series_id -> embedding row``. Index 0 is reserved for unseen series,
        #: so a checkpoint stays usable if the private split adds a new unit.
        self.series_index: dict[Any, int] = (
            dict(series_index)
            if series_index is not None
            else {key: position + 1 for position, key in enumerate(sorted(map(str, series)))}
        )

    def index_of(self, series_id: Any) -> int:
        """Embedding row of a series, ``0`` when it was not seen during training."""
        return int(self.series_index.get(str(series_id), 0))

    @property
    def num_series_slots(self) -> int:
        """Embedding table size, including the reserved unknown-series row."""
        return (max(self.series_index.values()) + 1) if self.series_index else 1

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, key: Any) -> SeriesArrays:
        return self.series[key]

    def keys(self) -> Iterable[Any]:
        return self.series.keys()

    def items(self) -> Iterable[tuple[Any, SeriesArrays]]:
        return self.series.items()

    @property
    def num_history_features(self) -> int:
        return self.spec.num_history_features

    @property
    def num_future_features(self) -> int:
        return self.spec.num_future_features

    @property
    def num_static_features(self) -> int:
        return self.spec.num_static_features

    @property
    def future_index(self) -> np.ndarray:
        return np.asarray(self.spec.future_index, dtype=int)

    @property
    def past_only_index(self) -> np.ndarray:
        return np.asarray(self.spec.past_only_index, dtype=int)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        spec: FeatureSpec,
        scaler: FeatureScaler | None = None,
        series_index: Mapping[Any, int] | None = None,
    ) -> "SeriesPanel":
        """Build a panel from a long table.

        Rows without a target (e.g. ``validation_input.csv``) are kept with
        ``y = NaN``; the rollout code fills those positions with predictions.
        Missing covariates are linearly interpolated inside each series and the
        original missingness is preserved as an explicit indicator feature.
        """
        work = frame.copy()
        work[spec.time_col] = pd.to_datetime(work[spec.time_col])
        work = work.sort_values([spec.series_col, spec.time_col], kind="mergesort")

        for col in spec.covariates + spec.static:
            if col not in work.columns:
                raise KeyError(f"Missing required covariate column {col!r}.")

        if scaler is None:
            scaler = FeatureScaler.fit(work, list(spec.covariates) + list(spec.static))

        series: dict[Any, SeriesArrays] = {}
        for series_id, group in work.groupby(spec.series_col, sort=False):
            columns: list[np.ndarray] = []
            for col in spec.covariates:
                values = pd.to_numeric(group[col], errors="coerce")
                filled = values.interpolate(limit_direction="both").fillna(scaler.median(col))
                columns.append(scaler.standardize(filled.to_numpy(dtype=np.float64), col))
            for col in spec.mask_features:
                columns.append(pd.to_numeric(group[col], errors="coerce").isna().to_numpy(dtype=np.float64))
            x = np.stack(columns, axis=-1).astype(np.float32) if columns else np.zeros((len(group), 0), np.float32)

            static_values = [
                float(scaler.standardize(np.asarray(pd.to_numeric(group[col], errors="coerce").iloc[0]), col))
                for col in spec.static
            ]
            static = np.asarray(static_values, dtype=np.float32)

            if spec.target_col in group.columns:
                y = pd.to_numeric(group[spec.target_col], errors="coerce").to_numpy(dtype=np.float32)
            else:
                y = np.full(len(group), np.nan, dtype=np.float32)

            series[series_id] = SeriesArrays(
                series_id=series_id,
                timestamps=group[spec.time_col].to_numpy(),
                y=y,
                x=x,
                static=static,
            )

        if series:
            width = next(iter(series.values())).x.shape[1]
            if width != spec.num_history_features:
                raise RuntimeError(
                    f"Built {width} feature columns but the spec declares {spec.num_history_features}."
                )
        return cls(series=series, spec=spec, scaler=scaler, series_index=series_index)


def concat_inputs(frames: Sequence[pd.DataFrame], spec: FeatureSpec) -> pd.DataFrame:
    """Glue several covariate tables into one long frame on a shared grid.

    Later frames win on duplicated ``(series_id, timestamp)`` keys so that an
    input directory supplied at evaluation time overrides bundled context.
    """
    usable = [f for f in frames if f is not None and len(f) > 0]
    if not usable:
        raise ValueError("No input tables provided.")
    prepared = []
    for order, frame in enumerate(usable):
        part = frame.copy()
        part[spec.time_col] = pd.to_datetime(part[spec.time_col])
        part["_source_order"] = order
        prepared.append(part)
    merged = pd.concat(prepared, ignore_index=True, sort=False)
    merged = merged.sort_values(
        [spec.series_col, spec.time_col, "_source_order"], kind="mergesort"
    )
    merged = merged.drop_duplicates([spec.series_col, spec.time_col], keep="last")
    return merged.drop(columns="_source_order").reset_index(drop=True)


def split_positions(length: int, val_len: int, test_len: int) -> tuple[slice, slice, slice]:
    """Chronological (fit, validation, test) position slices for local scoring.

    The public release has no validation labels, so all model selection happens
    on a held-out tail of ``train.csv``. ``test`` mimics the private split: it
    starts ``val_len`` steps after the last observed target, which is exactly
    the gap the private test window has relative to the released history.
    """
    if val_len + test_len >= length:
        raise ValueError("Split lengths exceed the available history.")
    fit_end = length - val_len - test_len
    return (
        slice(0, fit_end),
        slice(fit_end, fit_end + val_len),
        slice(fit_end + val_len, fit_end + val_len + test_len),
    )
