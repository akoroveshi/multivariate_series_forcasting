"""Blend the trained members into a single ensemble checkpoint.

The blending weight is a single scalar chosen by grid search. *Which* split it is
chosen on turns out to matter more than its value. Three evaluations are reported:

``val``
    forecast origin at the end of the fit slice -- the public-leaderboard regime.
``val_gap``
    the same validation labels, but reached across a 336-step unobserved gap
    (the origin is moved back to ``fit_end - 336``). This uses validation labels
    only, and it is the regime the private test split actually has.
``test``
    the held-out test window, scored once and never used for any decision.

The weight is selected on ``val_gap`` by default. Selecting it on ``val`` instead
overweights the recurrent member, whose error grows with the number of chained
blocks, and that error is invisible until the gap appears.

Usage::

    python scripts/build_ensemble.py --members runs/patchtst_main/checkpoint.pt \
        runs/lstm_main/checkpoint.pt --out results/ensemble.json
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

from src.evaluate import evaluate_split, labels_from_panel  # noqa: E402
from src.features import FeatureSpec, split_positions  # noqa: E402
from src.metrics import compute_metrics, format_metrics  # noqa: E402
from src.model import ForecastModel  # noqa: E402
from src.train import build_panel  # noqa: E402


def member_predictions(
    checkpoints: list[Path],
    panel,
    positions: slice,
    history_end: int,
    device: str,
) -> tuple[np.ndarray, list[np.ndarray], pd.DataFrame]:
    """Return ``(truth, [raw member predictions], index frame)`` aligned row-wise.

    Members are evaluated **unclipped** on purpose. :class:`EnsembleForecaster`
    blends raw member rollouts and the clip is applied once to the blend, so
    clipping here first would make the searched score a different function from
    the one the saved checkpoint computes.
    """
    labels = labels_from_panel(panel, positions)
    frames = []
    for path in checkpoints:
        model = ForecastModel.load_checkpoint(str(path), map_location=device).to(device)
        _, predictions = evaluate_split(
            model, panel, positions, history_end, device, model.history_len, clip_min=None
        )
        merged = labels[["series_id", "timestamp"]].merge(
            predictions, on=["series_id", "timestamp"], how="left", validate="one_to_one"
        )
        frames.append(merged["prediction"].to_numpy())
    return labels["target"].to_numpy(), frames, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Search blending weights and save an ensemble.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument(
        "--members",
        nargs="+",
        type=Path,
        default=[
            ROOT / "runs" / "patchtst_main" / "checkpoint.pt",
            ROOT / "runs" / "lstm_main" / "checkpoint.pt",
        ],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ensemble.json")
    parser.add_argument("--checkpoint-out", type=Path, default=ROOT / "runs" / "ensemble" / "checkpoint.pt")
    parser.add_argument("--val-len", type=int, default=336)
    parser.add_argument("--test-len", type=int, default=336)
    parser.add_argument(
        "--select-on",
        default="val_gap",
        choices=["val", "val_gap"],
        help="Which validation-label evaluation selects the blending weight.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    missing = [str(p) for p in args.members if not p.exists()]
    if missing:
        raise SystemExit(f"Missing member checkpoint(s): {missing}")
    if len(args.members) != 2:
        raise SystemExit("The weight grid search is implemented for exactly two members.")

    frame = pd.read_csv(args.train)
    length = int(frame.groupby("series_id", sort=False).size().max())
    fit_end = length - args.val_len - args.test_len
    panel, length = build_panel(frame, FeatureSpec.benchmark(), fit_end)
    fit_slice, val_slice, test_slice = split_positions(length, args.val_len, args.test_len)

    loaded_configs = [
        ForecastModel.load_checkpoint(str(p), map_location=args.device).config for p in args.members
    ]
    clip_min = min(float(cfg.get("clip_min", 0.0)) for cfg in loaded_configs)

    results: dict[str, object] = {"members": [str(p) for p in args.members], "clip_min": clip_min}
    grids: dict[str, list[dict[str, float]]] = {}
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    best_weight = 1.0

    # (name, scored positions, forecast origin). ``val_gap`` reuses the validation
    # labels but moves the origin back by one validation window, so the members are
    # exercised over the same unobserved gap the private split has.
    evaluations = [
        ("val", val_slice, fit_slice.stop),
        ("val_gap", val_slice, fit_slice.stop - args.val_len),
        ("test", test_slice, fit_slice.stop),
    ]
    order = {name: index for index, (name, _, _) in enumerate(evaluations)}
    evaluations.sort(key=lambda item: item[0] != args.select_on)

    for split_name, positions, origin in evaluations:
        truth, member_preds, _ = member_predictions(
            args.members, panel, positions, origin, args.device
        )
        grid = []
        for weight in weights:
            # Blend first, clip once -- exactly what EnsembleForecaster.rollout does.
            blended = np.maximum(
                weight * member_preds[0] + (1.0 - weight) * member_preds[1], clip_min
            )
            metrics = compute_metrics(truth, blended)
            grid.append({"weight": float(weight), **metrics})
        grids[split_name] = grid
        if split_name == args.select_on:
            best = min(grid, key=lambda row: row["mae"])
            best_weight = float(best["weight"])
            print(f"selected weight on '{split_name}': {best_weight:.2f} (WAPE {best['wape']:.3f})")
        chosen = next(row for row in grid if abs(row["weight"] - best_weight) < 1e-9)
        results[split_name] = {k: v for k, v in chosen.items() if k != "weight"}
        results[f"{split_name}_members"] = [
            compute_metrics(truth, np.maximum(preds, clip_min)) for preds in member_preds
        ]
        print(f"[{split_name:7s}] ensemble {format_metrics(results[split_name])}")
        for name, metrics in zip(args.members, results[f"{split_name}_members"]):
            print(f"[{split_name:7s}] {name.parent.name:16s} {format_metrics(metrics)}")

    results["weight"] = best_weight
    results["selected_on"] = args.select_on
    results["weight_grid"] = {k: grids[k] for k in sorted(grids, key=lambda n: order[n])}

    # Record what the *other* criterion would have picked. This is the interesting
    # comparison: a weight tuned on the leaderboard regime looks better there and
    # is worse where it actually has to run.
    other = "val" if args.select_on == "val_gap" else "val_gap"
    alternative_weight = float(min(grids[other], key=lambda row: row["mae"])["weight"])
    results["alternative"] = {
        "selected_on": other,
        "weight": alternative_weight,
        **{
            name: {
                k: v
                for k, v in next(
                    row for row in grid if abs(row["weight"] - alternative_weight) < 1e-9
                ).items()
                if k != "weight"
            }
            for name, grid in grids.items()
        },
    }
    print(
        f"[compare] selecting on '{other}' would pick w={alternative_weight:.2f}: "
        f"val WAPE {results['alternative']['val']['wape']:.3f}, "
        f"test WAPE {results['alternative']['test']['wape']:.3f} "
        f"(chosen w={best_weight:.2f}: {results['val']['wape']:.3f} / {results['test']['wape']:.3f})"
    )

    # Persist the ensemble as one checkpoint so predict.py stays unchanged.
    loaded = [ForecastModel.load_checkpoint(str(p), map_location=args.device) for p in args.members]
    ensemble = ForecastModel(
        model_type="ensemble",
        members=[m.config for m in loaded],
        member_weights=[best_weight, 1.0 - best_weight],
        history_len=max(m.history_len for m in loaded),
        clip_min=clip_min,
    )
    for slot, member in zip(ensemble.net.members, loaded):
        slot.load_state_dict(member.net.state_dict())
    ensemble.attach_preprocessing(loaded[0].feature_spec, loaded[0].feature_scaler)
    ensemble.attach_series_index(loaded[0].series_index)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save_checkpoint(str(args.checkpoint_out))

    # Sanity check: the saved artefact must reproduce the searched score.
    reloaded = ForecastModel.load_checkpoint(str(args.checkpoint_out), map_location=args.device)
    verify, _ = evaluate_split(
        reloaded,
        panel,
        val_slice,
        fit_slice.stop,
        args.device,
        reloaded.history_len,
        clip_min=reloaded.config.get("clip_min"),
    )
    results["checkpoint_val"] = verify
    print(f"[verify] reloaded ensemble {format_metrics(verify)}")
    searched = results["val"]["wape"]
    if abs(verify["wape"] - searched) > 1e-4:
        raise SystemExit(
            f"Saved checkpoint scores {verify['wape']:.4f} WAPE on 'val' but the weight "
            f"search reported {searched:.4f}: the searched blend and the serialised one "
            "disagree."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out} and {args.checkpoint_out}")


if __name__ == "__main__":
    main()
