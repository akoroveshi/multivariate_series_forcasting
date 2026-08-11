"""Generate the report figures from the trained checkpoints and result JSONs.

Figures written to ``report/figures``:

``error_by_horizon.pdf`` (embedded in the report)
    MAE against the forecast step across both held-out regimes, with the
    pre-RevIN-repair recurrent model as the contrast. Also prints the fitted
    drift per 100 steps, which is the number the report quotes.
``permutation_importance.pdf``
    Increase in validation WAPE when one known-future covariate is shuffled
    across the forecast window. Quantifies which drivers the model actually uses;
    the report quotes the top four and points here for the rest.
``forecast_examples.pdf``
    Predicted versus observed load for the median unit over the validation window.
``ablations.pdf``
    Validation and test WAPE of every ablation, sorted by validation WAPE --- a
    quick-look companion to Table 2 of the report.

The last three are repository artefacts referenced by filename from the report
rather than embedded, which is what keeps the report inside its page limit.

Usage::

    python scripts/make_figures.py
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.baselines import make_all_baselines  # noqa: E402
from src.evaluate import evaluate_split, labels_from_panel  # noqa: E402
from src.features import FeatureSpec, split_positions  # noqa: E402
from src.model import ForecastModel  # noqa: E402
from src.train import build_panel  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from variants import checkpoint_for, select_best  # noqa: E402

# Categorical slots 1-3 of the validated palette (they clear the all-pairs CVD
# floors); baselines use chart ink so only the models carry hue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
        "lines.linewidth": 1.2,
    }
)


def clean(axes) -> None:
    """Recessive chrome: drop the top/right spines, keep a hairline grid."""
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


def model_predictions(
    checkpoint: Path, panel, positions: slice, history_end: int, device: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Predictions for a positional slice, aligned to ``labels_from_panel`` order."""
    model = ForecastModel.load_checkpoint(str(checkpoint), map_location=device).to(device)
    labels = labels_from_panel(panel, positions)
    _, frame = evaluate_split(
        model, panel, positions, history_end, device, model.history_len,
        clip_min=model.config.get("clip_min"),
    )
    merged = labels[["series_id", "timestamp"]].merge(
        frame, on=["series_id", "timestamp"], how="left", validate="one_to_one"
    )
    return merged["prediction"].to_numpy(), labels


def baseline_predictions(panel, fit_slice: slice, positions: slice, name: str) -> np.ndarray:
    """Reference baseline aligned to ``labels_from_panel`` order."""
    spec = panel.spec
    rows = []
    for series_id, arrays in panel.items():
        rows.append(
            pd.DataFrame(
                {
                    spec.series_col: series_id,
                    spec.time_col: arrays.timestamps[fit_slice],
                    spec.target_col: arrays.y[fit_slice],
                }
            )
        )
    fit_frame = pd.concat(rows, ignore_index=True)
    labels = labels_from_panel(panel, positions)
    predictions = make_all_baselines(fit_frame, labels[[spec.series_col, spec.time_col]])[name]
    merged = labels[[spec.series_col, spec.time_col]].merge(
        predictions, on=[spec.series_col, spec.time_col], how="left", validate="one_to_one"
    )
    return merged["prediction"].to_numpy()


def figure_error_by_horizon(panel, checkpoints, fit_slice, val_slice, test_slice, device, out: Path):
    """MAE per forecast step, as one continuous curve across both regimes.

    The two held-out windows are contiguous and share a forecast origin, so steps
    1--336 are the validation regime and 337--672 the test regime. Plotting them as
    one curve (rather than two panels restarting at $h=1$) is what makes error
    accumulation visible at all.
    """
    positions = slice(val_slice.start, test_slice.stop)
    labels = labels_from_panel(panel, positions)
    # ``to_numpy()`` can hand back a read-only view depending on the pandas/numpy
    # build, so add out-of-place rather than in-place.
    steps = labels.groupby("series_id", sort=False).cumcount().to_numpy()
    steps = steps + (positions.start - fit_slice.stop)
    truth = labels["target"].to_numpy()

    series = []
    for name, (path, color, style) in checkpoints.items():
        if not path.exists():
            print(f"  (skipping {name}: {path} not found)")
            continue
        predictions, _ = model_predictions(path, panel, positions, fit_slice.stop, device)
        series.append((name, color, style, predictions))
    # No baseline curve here on purpose: the seasonal mean sits 2-3x higher and
    # would compress the models into the bottom fifth of the axis, which is exactly
    # the range this figure exists to compare. Baselines are in the results table.

    figure, axes = plt.subplots(figsize=(3.3, 2.15))
    boundary = val_slice.stop - fit_slice.stop
    axes.axvline(boundary, color=AXIS, linewidth=0.7, linestyle=":")
    slopes: dict[str, float] = {}
    for name, color, style, predictions in series:
        curve = pd.Series(np.abs(truth - predictions)).groupby(steps).mean()
        smooth = curve.rolling(12, min_periods=1, center=True).mean()
        axes.plot(curve.index + 1, smooth.to_numpy(), color=color, linestyle=style, label=name)
        # Drift per 100 forecast steps: the quantity the figure exists to compare.
        slopes[name] = float(np.polyfit(curve.index.to_numpy(), curve.to_numpy(), 1)[0] * 100)
    print("  MAE drift per 100 steps: " + ", ".join(f"{k}={v:+.3f}" for k, v in slopes.items()))
    # The regime boundary is annotated on the axis rather than inside the plot, so
    # the legend can sit in the empty upper-left without colliding with either.
    axes.annotate(
        "gap begins", xy=(boundary, axes.get_ylim()[0]), xytext=(boundary + 8, axes.get_ylim()[0]),
        fontsize=6.5, color=MUTED, va="bottom", ha="left",
    )
    axes.set_xlabel("forecast step $h$ (hours after the origin)")
    axes.set_ylabel("MAE (12 h moving avg.)")
    axes.set_xlim(1, steps.max() + 1)
    axes.legend(loc="upper left", fontsize=6.5)
    clean(axes)
    figure.savefig(out)
    plt.close(figure)
    print("wrote", out)


def figure_permutation_importance(panel, checkpoint: Path, fit_slice, val_slice, device, out: Path, top: int = 12):
    """Validation WAPE increase when a known-future covariate is shuffled."""
    model = ForecastModel.load_checkpoint(str(checkpoint), map_location=device).to(device)
    spec = panel.spec
    base, _ = evaluate_split(
        model, panel, val_slice, fit_slice.stop, device, model.history_len,
        clip_min=model.config.get("clip_min"),
    )

    rng = np.random.default_rng(0)
    deltas: dict[str, float] = {}
    originals = {sid: arrays.x for sid, arrays in panel.items()}
    for column, name in enumerate(spec.covariates):
        for sid, arrays in panel.items():
            shuffled = originals[sid].copy()
            block = shuffled[val_slice, column]
            shuffled[val_slice, column] = rng.permutation(block)
            arrays.x = shuffled
        metrics, _ = evaluate_split(
            model, panel, val_slice, fit_slice.stop, device, model.history_len,
            clip_min=model.config.get("clip_min"),
        )
        deltas[name] = metrics["wape"] - base["wape"]
        for sid, arrays in panel.items():
            arrays.x = originals[sid]

    ranked = sorted(deltas.items(), key=lambda item: item[1], reverse=True)[:top][::-1]
    figure, axes = plt.subplots(figsize=(3.3, 0.21 * len(ranked) + 0.7))
    positions = np.arange(len(ranked))
    axes.barh(positions, [value for _, value in ranked], color=BLUE, height=0.62)
    axes.set_yticks(positions, [name.replace("_", " ") for name, _ in ranked])
    axes.set_xlabel("$\\Delta$ validation WAPE (pp) when shuffled")
    axes.grid(axis="y", visible=False)
    clean(axes)
    figure.savefig(out)
    plt.close(figure)
    print("wrote", out, f"(base WAPE {base['wape']:.2f})")
    return {"base_wape": base["wape"], "deltas": deltas}


def figure_forecast_examples(panel, checkpoint: Path, fit_slice, val_slice, device, out: Path):
    """Observed versus predicted load for three representative units."""
    predictions, labels = model_predictions(checkpoint, panel, val_slice, fit_slice.stop, device)
    seasonal = baseline_predictions(panel, fit_slice, val_slice, "seasonal_mean")
    labels = labels.assign(prediction=predictions, seasonal=seasonal)
    error = (
        labels.assign(err=(labels.target - labels.prediction).abs())
        .groupby("series_id")
        .err.mean()
        .sort_values()
    )
    # One panel keeps the figure inside the report's page budget; the median unit
    # is the honest choice (best flatters, worst caricatures).
    picks = [error.index[len(error) // 2]]
    titles = ["median unit by MAE"]

    figure, axes_column = plt.subplots(
        len(picks), 1, figsize=(3.3, 1.5 * len(picks) + 0.35), sharex=True, squeeze=False
    )
    axes_column = axes_column.ravel()
    for axes, series_id, rank in zip(axes_column, picks, titles):
        part = labels.loc[labels.series_id.eq(series_id)].sort_values("timestamp")
        hours = np.arange(len(part))
        axes.plot(hours, part.target, color=INK, linewidth=0.9, label="observed")
        axes.plot(hours, part.prediction, color=BLUE, label="model")
        axes.plot(hours, part.seasonal, color=MUTED, linestyle="--", label="seasonal mean")
        axes.set_ylabel("load index")
        axes.set_title(
            f"{rank}: {series_id}  (MAE {np.abs(part.target - part.prediction).mean():.2f})",
            loc="left",
        )
        clean(axes)
    axes_column[-1].set_xlabel("hours into the 336-step validation window")
    axes_column[0].legend(loc="upper right", ncol=3)
    figure.savefig(out)
    plt.close(figure)
    print("wrote", out)


def figure_ablations(path: Path, out: Path):
    """Validation and test WAPE for each ablation row."""
    if not path.exists():
        print("skipping ablation figure:", path, "not found")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        (name, entry["val"]["wape"], entry["test"]["wape"])
        for name, entry in data.items()
        if entry.get("val") and entry.get("test")
    ]
    rows.sort(key=lambda row: row[1], reverse=True)
    figure, axes = plt.subplots(figsize=(6.6, 0.26 * len(rows) + 0.8))
    positions = np.arange(len(rows))
    axes.barh(positions - 0.19, [row[1] for row in rows], height=0.36, color=BLUE, label="validation")
    axes.barh(positions + 0.19, [row[2] for row in rows], height=0.36, color=ORANGE, label="test (gap)")
    axes.set_yticks(positions, [row[0].replace("_", " ") for row in rows])
    axes.set_xlabel("WAPE (%), lower is better")
    axes.grid(axis="y", visible=False)
    axes.legend(loc="lower right", ncol=2)
    clean(axes)
    figure.savefig(out)
    plt.close(figure)
    print("wrote", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the report figures.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "report" / "figures")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--val-len", type=int, default=336)
    parser.add_argument("--test-len", type=int, default=336)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    chosen = select_best(args.results_dir)
    # Identity is carried by hue (model family) and the variant by line style, so
    # the two LSTM curves read as the same model before and after the RevIN repair.
    checkpoints = {
        "PatchTST-cov": (checkpoint_for(chosen.get("patchtst", "patchtst_main"), ROOT), ORANGE, "-"),
        "LSTM+Attention": (checkpoint_for(chosen.get("lstm_attention", "lstm_main"), ROOT), AQUA, "-"),
        "LSTM+Attention, RevIN": (ROOT / "runs" / "lstm_main" / "checkpoint.pt", AQUA, "--"),
        "Blend": (ROOT / "runs" / "ensemble" / "checkpoint.pt", BLUE, "-"),
    }
    available = {name: spec for name, spec in checkpoints.items() if spec[0].exists()}
    if not available:
        raise SystemExit("No checkpoints found; run scripts/run_all.sh first.")
    print("using checkpoints:", ", ".join(available))

    frame = pd.read_csv(args.train)
    length = int(frame.groupby("series_id", sort=False).size().max())
    panel, length = build_panel(frame, FeatureSpec.benchmark(), length - args.val_len - args.test_len)
    fit_slice, val_slice, test_slice = split_positions(length, args.val_len, args.test_len)

    figure_error_by_horizon(
        panel, available, fit_slice, val_slice, test_slice, args.device,
        args.figures_dir / "error_by_horizon.pdf",
    )
    # The analysis figures use the single best architecture rather than the blend:
    # attributing a permutation effect to a mixture of two decoders is not
    # interpretable, and the blend is degenerate anyway once tuned for the gap.
    primary = (available.get("PatchTST-cov") or next(iter(available.values())))[0]
    importance = figure_permutation_importance(
        panel, primary, fit_slice, val_slice, args.device,
        args.figures_dir / "permutation_importance.pdf",
    )
    (args.results_dir / "permutation_importance.json").write_text(
        json.dumps(importance, indent=2, default=float), encoding="utf-8"
    )
    figure_forecast_examples(
        panel, primary, fit_slice, val_slice, args.device,
        args.figures_dir / "forecast_examples.pdf",
    )
    figure_ablations(args.results_dir / "ablations.json", args.figures_dir / "ablations.pdf")


if __name__ == "__main__":
    main()
