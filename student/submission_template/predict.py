"""Inference entrypoint for final private evaluation.

Loads a trained checkpoint and rolls it forward, per series, to cover
every timestamp in the provided forecast index. The 24-hour rollout
block is chained autoregressively (each predicted block is fed back as
additional history) until the full 336-step benchmark horizon is covered.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.dataset import hours_since_epoch
from src.model import ForecastModel
from src.models.common import calendar_features

DEFAULT_HISTORY_LEN = 168


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    """ Load the rows that need predictions """
    candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    
    for forecast_index in candidates:
        if forecast_index.exists():
            return pd.read_csv(forecast_index)
    expected = ", ".join(path.name for path in candidates)
    
    raise FileNotFoundError(f"Expected one of {expected} in input_dir.")


def load_history(input_dir: Path) -> pd.DataFrame:
    """ Load the history table used as conditioning context for every series """
    history_path = input_dir / "train.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Expected train.csv (history) in {input_dir}.")
    return pd.read_csv(history_path)


def predict_series(
    model: ForecastModel,
    history: pd.DataFrame,
    index_part: pd.DataFrame,
    device: torch.device,
    history_len: int,
    time_col: str,
    target_col: str,
) -> np.ndarray:
    """ Roll the model forward to cover every timestamp requested """
    history = history.sort_values(time_col).tail(history_len)
    if len(history) == 0:
        raise ValueError("No history available for a required series.")
    if len(history) < history_len:
        # Not enough context yet, repeat the earliest
        # value backward, encoder still receives history_len steps
        pad = history_len - len(history)
        first_row = history.iloc[[0]]
        history = pd.concat([pd.concat([first_row] * pad, ignore_index=True), history], ignore_index=True)

    hist_target = torch.tensor(
        history[target_col].to_numpy(dtype=np.float32), device=device
    ).view(1, -1, 1)
    hist_hours = torch.tensor(hours_since_epoch(history[time_col]), device=device)
    hist_calendar = calendar_features(hist_hours).unsqueeze(0)

    index_part = index_part.sort_values(time_col)
    future_hours = torch.tensor(hours_since_epoch(index_part[time_col]), device=device)
    decoder_calendar = calendar_features(future_hours).unsqueeze(0)

    predictions = model.rollout(hist_target, hist_calendar, decoder_calendar, horizon=len(index_part))
    return predictions.squeeze(0).squeeze(-1).cpu().numpy()


def main() -> None:
    """ Load a checkpoint and write predictions for every row """
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--history_len", type=int, default=DEFAULT_HISTORY_LEN, help="Trailing history steps fed to the encoder")
    parser.add_argument("--series_col", default="series_id")
    parser.add_argument("--time_col", default="timestamp")
    parser.add_argument("--target_col", default="target")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastModel.load_checkpoint(str(args.checkpoint), map_location=str(device)).to(device)
    model.eval()

    forecast_index = load_forecast_index(args.input_dir)
    history = load_history(args.input_dir)

    rows = []
    for series_id, index_part in forecast_index.groupby(args.series_col, sort=False):
        series_history = history.loc[history[args.series_col].eq(series_id)]
        preds = predict_series(
            model,
            series_history,
            index_part,
            device,
            args.history_len,
            args.time_col,
            args.target_col,
        )
        result = index_part.sort_values(args.time_col)[[args.series_col, args.time_col]].copy()
        result["prediction"] = preds
        rows.append(result)

    predictions = pd.concat(rows, ignore_index=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()