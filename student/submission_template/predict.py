"""Inference entrypoint for the private evaluation.

Required command::

    python predict.py --input_dir /data/input --output_file /output/predictions.csv \
        --checkpoint /submission/checkpoint.pt

The private input directory contains ``test_input.csv`` (covariates for the test
window), ``forecast_index_test.csv`` and ``metadata.json`` -- but no observed
target. The last released target therefore sits 336 steps before the first test
timestamp. This script reconstructs a continuous per-series timeline from three
sources, in increasing priority:

1. the **context table** bundled inside the checkpoint (tail of the public
   history *with* targets, plus the public validation covariates),
2. any table found in ``input_dir`` (``test_input.csv``, ``validation_input.csv``,
   ``train.csv``, ``history.csv``),
3. observed targets, wherever any of those tables provide them.

The model is then rolled forward from the last observed target to the final
requested timestamp and only the rows of the forecast index are written out. No
network access is required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.evaluate import forecast_frame
from src.features import FeatureSpec, SeriesPanel, concat_inputs
from src.model import ForecastModel

FORECAST_INDEX_NAMES = ("forecast_index_test.csv", "forecast_index_validation.csv", "forecast_index.csv")
INPUT_TABLE_NAMES = ("train.csv", "history.csv", "validation_input.csv", "test_input.csv")


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    """Load the rows that need predictions."""
    for name in FORECAST_INDEX_NAMES:
        candidate = input_dir / name
        if candidate.exists():
            return pd.read_csv(candidate)
    expected = ", ".join(FORECAST_INDEX_NAMES)
    raise FileNotFoundError(f"Expected one of {expected} in {input_dir}.")


def load_input_tables(input_dir: Path) -> list[pd.DataFrame]:
    """Load every covariate/history table present in the input directory."""
    tables = []
    for name in INPUT_TABLE_NAMES:
        candidate = input_dir / name
        if candidate.exists():
            tables.append(pd.read_csv(candidate))
    return tables


def build_panel(
    model: ForecastModel, input_dir: Path, forecast_index: pd.DataFrame
) -> SeriesPanel:
    """Glue bundled context and provided tables into one panel."""
    spec = model.feature_spec or FeatureSpec.benchmark()
    tables: list[pd.DataFrame] = []
    if model.context is not None:
        tables.append(model.context)
    tables.extend(load_input_tables(input_dir))
    if not tables:
        raise FileNotFoundError(
            "No covariate table available: the checkpoint carries no context and "
            f"{input_dir} contains none of {INPUT_TABLE_NAMES}."
        )

    merged = concat_inputs(tables, spec)
    # Compare on parsed timestamps, not on their string form: the private tables
    # need not use the same textual datetime format as the bundled context.
    def keys(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
        stamps = pd.to_datetime(frame[spec.time_col])
        return set(zip(frame[spec.series_col].astype(str), stamps))

    missing = keys(forecast_index) - keys(merged)
    if missing:
        raise ValueError(
            f"{len(missing)} forecast-index rows have no covariates in the provided tables "
            f"(first: {sorted(missing)[0]})."
        )
    if spec.target_col not in merged.columns:
        merged[spec.target_col] = float("nan")
    return SeriesPanel.from_frame(
        merged, spec, model.feature_scaler, series_index=model.series_index
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate forecasts for a forecast index.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = torch.device(args.device)
    model = ForecastModel.load_checkpoint(str(args.checkpoint), map_location=str(device)).to(device)
    model.eval()

    forecast_index = load_forecast_index(args.input_dir)
    panel = build_panel(model, args.input_dir, forecast_index)

    predictions = forecast_frame(
        model,
        panel,
        forecast_index,
        device=device,
        history_len=model.history_len,
        batch_size=args.batch_size,
        clip_min=model.config.get("clip_min"),
    )

    spec = panel.spec
    ordered = forecast_index[[spec.series_col, spec.time_col]].copy()
    ordered[spec.time_col] = pd.to_datetime(ordered[spec.time_col])
    ordered = ordered.merge(predictions, on=[spec.series_col, spec.time_col], how="left")
    if ordered["prediction"].isna().any():
        raise RuntimeError(f"{int(ordered['prediction'].isna().sum())} forecast rows were not filled.")
    ordered[spec.time_col] = forecast_index[spec.time_col].to_numpy()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(args.output_file, index=False)
    print(f"wrote {len(ordered):,} predictions to {args.output_file}")


if __name__ == "__main__":
    main()
