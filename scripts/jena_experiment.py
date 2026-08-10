"""Additional dataset experiment: Jena Climate (Max Planck Institute for Biogeochemistry).

Dataset
-------
Ten-minute recordings of 14 atmospheric variables at the MPI-BGC weather station
in Jena, Germany, from 2009-01-01 to 2017-01-01 (420,551 rows). We resample to
hourly means to match the benchmark's sampling rate, convert the wind
direction/speed pair into Cartesian components, and repair the ``-9999`` sentinel
values in the wind channels.

Why it is a useful stress test
------------------------------
The benchmark is an *exogenous-covariate* problem: all 22 covariates are known
for the forecast window, so most of the signal is available at prediction time.
Jena is the opposite regime -- the classical long-term forecasting setting in
which every channel except the calendar encodings is **past-only**. Running the
same architecture on it therefore tests whether the gains come from the modelling
or merely from the covariate contract.

Protocol
--------
Chronological 70/10/20 split, history 168 h, horizon 336 h, rolling-origin
evaluation with a 336-step stride. Temperature crosses zero, so WAPE/MAPE are not
meaningful; MAE and RMSE are the reported metrics.

Usage::

    python scripts/jena_experiment.py --models patchtst lstm_attention --targets "T (degC)"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student" / "submission_template"))

from src.features import FeatureScaler, FeatureSpec, SeriesPanel, derive_calendar_features, split_positions  # noqa: E402
from src.metrics import compute_metrics, format_metrics  # noqa: E402
from src.train import MODEL_DEFAULTS, parse_args, run_experiment  # noqa: E402

RAW_COLUMNS = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
]
WIND_COLUMNS = ["wx", "wy", "max_wx", "max_wy"]
CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend", "trend")

CLEAN_NAMES = {
    "p (mbar)": "pressure",
    "T (degC)": "temperature",
    "Tpot (K)": "pot_temperature",
    "Tdew (degC)": "dew_point",
    "rh (%)": "humidity",
    "VPmax (mbar)": "vp_max",
    "VPact (mbar)": "vp_act",
    "VPdef (mbar)": "vp_def",
    "sh (g/kg)": "specific_humidity",
    "H2OC (mmol/mol)": "h2o_concentration",
    "rho (g/m**3)": "air_density",
    "wv (m/s)": "wind_speed",
    "max. wv (m/s)": "wind_speed_max",
}


def load_hourly(path: Path) -> pd.DataFrame:
    """Load the raw CSV and return an hourly, cleaned, renamed frame."""
    raw = pd.read_csv(path)
    stamps = pd.to_datetime(raw["Date Time"], format="%d.%m.%Y %H:%M:%S")
    frame = raw[RAW_COLUMNS + ["wd (deg)"]].copy()
    frame["timestamp"] = stamps
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")

    # -9999 is the station's missing-value sentinel in the wind channels.
    for col in ("wv (m/s)", "max. wv (m/s)"):
        frame.loc[frame[col] <= -9990, col] = np.nan
    frame[["wv (m/s)", "max. wv (m/s)"]] = frame[["wv (m/s)", "max. wv (m/s)"]].interpolate(
        limit_direction="both"
    )

    # Wind direction is circular: a plain hourly mean of degrees is meaningless.
    radians = np.deg2rad(frame["wd (deg)"].to_numpy())
    frame["wx"] = frame["wv (m/s)"] * np.cos(radians)
    frame["wy"] = frame["wv (m/s)"] * np.sin(radians)
    frame["max_wx"] = frame["max. wv (m/s)"] * np.cos(radians)
    frame["max_wy"] = frame["max. wv (m/s)"] * np.sin(radians)
    frame = frame.drop(columns=["wd (deg)"])

    hourly = (
        frame.set_index("timestamp")
        .resample("1h")
        .mean()
        .interpolate(limit_direction="both")
        .reset_index()
    )
    hourly = hourly.rename(columns=CLEAN_NAMES)
    hourly["series_id"] = "jena"
    return derive_calendar_features(hourly, time_col="timestamp")


def build_spec(target: str) -> FeatureSpec:
    """Calendar covariates are known ahead; every climate channel is past-only."""
    channels = [CLEAN_NAMES[c] for c in RAW_COLUMNS] + WIND_COLUMNS
    past_only = tuple(c for c in channels if c != target)
    return FeatureSpec(
        dynamic=CALENDAR,
        past_only=past_only,
        static=(),
        mask_features=(),
        target_col=target,
    )


def windowed_baselines(
    panel: SeriesPanel,
    positions: slice,
    history_len: int,
    horizon: int,
    fit_end: int,
) -> dict[str, dict[str, float]]:
    """Reference forecasts under the same rolling-origin protocol as the models."""
    arrays = next(iter(panel.series.values()))
    y = arrays.y
    stamps = pd.to_datetime(arrays.timestamps)
    hour = stamps.hour.to_numpy()
    dow = stamps.dayofweek.to_numpy()

    fit_frame = pd.DataFrame({"y": y[:fit_end], "hour": hour[:fit_end], "dow": dow[:fit_end]})
    seasonal = fit_frame.groupby(["dow", "hour"])["y"].mean()
    seasonal_lookup = np.array(
        [seasonal.get((d, h), float(fit_frame["y"].mean())) for d, h in zip(dow, hour)]
    )

    truths: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {
        "naive_last_value": [],
        "lag24_repeat": [],
        "seasonal_mean": [],
    }
    for anchor in range(positions.start, positions.stop - horizon + 1, horizon):
        truths.append(y[anchor : anchor + horizon])
        predictions["naive_last_value"].append(np.repeat(y[anchor - 1], horizon))
        last_day = y[anchor - 24 : anchor]
        predictions["lag24_repeat"].append(np.tile(last_day, int(np.ceil(horizon / 24)))[:horizon])
        predictions["seasonal_mean"].append(seasonal_lookup[anchor : anchor + horizon])

    truth = np.concatenate(truths)
    return {
        name: compute_metrics(truth, np.concatenate(values)) for name, values in predictions.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the architecture on the Jena Climate dataset.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "jena_climate_2009_2016.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "jena.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "jena")
    parser.add_argument("--models", nargs="+", default=["patchtst", "lstm_attention"])
    parser.add_argument("--targets", nargs="+", default=["temperature", "pressure"])
    parser.add_argument("--history-len", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=336)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hourly = load_hourly(args.csv)
    length = len(hourly)
    val_len = int(round(0.1 * length))
    test_len = int(round(0.2 * length))
    fit_end = length - val_len - test_len
    print(
        f"Jena hourly steps={length:,}  fit={fit_end:,}  val={val_len:,}  test={test_len:,}  "
        f"range={hourly.timestamp.min()} .. {hourly.timestamp.max()}"
    )

    results: dict[str, dict] = {}
    for target in args.targets:
        spec = build_spec(target)
        scaler = FeatureScaler.fit(hourly.iloc[:fit_end], list(spec.covariates))
        panel = SeriesPanel.from_frame(hourly, spec, scaler)
        fit_slice, val_slice, test_slice = split_positions(length, val_len, test_len)
        print(
            f"\n=== target={target}  history_features={panel.num_history_features} "
            f"future_features={panel.num_future_features} ==="
        )

        target_results: dict[str, dict] = {"baselines": {}}
        for split_name, positions in (("val", val_slice), ("test", test_slice)):
            target_results["baselines"][split_name] = windowed_baselines(
                panel, positions, args.history_len, args.horizon, fit_end
            )
            for name, metrics in target_results["baselines"][split_name].items():
                print(f"  [{split_name}] {name:18s} {format_metrics(metrics)}")

        for model_name in args.models:
            tag = f"jena_{target}_{model_name}"
            block_len = args.horizon if model_name == "patchtst" else MODEL_DEFAULTS[model_name]["block_len"]
            argv = [
                "--train", str(args.csv),  # unused: the panel is passed in directly
                "--checkpoint-out", str(args.runs_dir / tag / "checkpoint.pt"),
                "--metrics-out", str(args.runs_dir / tag / "metrics.json"),
                "--model", model_name,
                "--history-len", str(args.history_len),
                "--block-len", str(block_len),
                "--stride", str(args.stride),
                "--val-len", str(val_len),
                "--test-len", str(test_len),
                "--epochs", str(args.epochs),
                "--eval-protocol", "windows",
                "--eval-horizon", str(args.horizon),
                "--eval-stride", str(args.horizon),
                "--patience", "5",
                "--sgdr-t0", "5",
                "--sgdr-tmult", "1",
                "--seed", str(args.seed),
                "--tag", tag,
            ]
            print(f"\n--- training {tag} ---")
            summary = run_experiment(panel, parse_args(argv))
            target_results[model_name] = {
                "val": summary.get("val"),
                "test": summary.get("test"),
                "params": summary["params"],
                "best_epoch": summary["best_epoch"],
                "runtime_seconds": summary["runtime_seconds"],
                "config": summary["config"],
            }
        results[target] = target_results

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
