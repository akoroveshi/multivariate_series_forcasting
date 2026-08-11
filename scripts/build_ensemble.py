"""Blend trained runs into a single ensemble checkpoint.

Weights come from **greedy forward selection with replacement** (Caruana et al.'s
ensemble selection): start from the empty blend and repeatedly add whichever member
most improves the criterion, allowing the same member to be picked again. After
``--rounds`` picks the weights are the pick counts, normalised. This scales to any
number of members, needs no combinatorial search, and resists overfitting a small
validation window far better than optimising a continuous weight vector.

*Which* split the weights are chosen on matters more than their values, so three
evaluations are reported:

``val``
    forecast origin at the end of the fit slice -- the public-leaderboard regime.
``val_gap``
    the same validation labels, but reached across a 336-step unobserved gap (the
    origin is moved back one window). This is the regime the private split has.
``test``
    the held-out test window, scored once and never used for any decision.

Members are evaluated **unclipped**: :class:`EnsembleForecaster` blends raw member
rollouts and applies the clip once to the blend, so clipping first would make the
searched score a different function from the one the saved checkpoint computes.

Usage::

    python scripts/build_ensemble.py --members runs/a/checkpoint.pt runs/b/checkpoint.pt ...
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
    checkpoints: list[Path], panel, positions: slice, history_end: int, device: str
) -> tuple[np.ndarray, list[np.ndarray]]:
    """``(truth, [raw per-member predictions])``, aligned row-wise."""
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
    return labels["target"].to_numpy(), frames


def greedy_weights(
    truth: np.ndarray, preds: list[np.ndarray], clip_min: float, rounds: int
) -> tuple[np.ndarray, list[str]]:
    """Greedy forward selection with replacement; returns normalised weights."""
    counts = np.zeros(len(preds))
    running = np.zeros_like(truth, dtype=float)
    trace = []
    for step in range(1, rounds + 1):
        best, best_mae = None, np.inf
        for index, candidate in enumerate(preds):
            blended = np.maximum((running * (step - 1) + candidate) / step, clip_min)
            mae = float(np.mean(np.abs(truth - blended)))
            if mae < best_mae:
                best, best_mae = index, mae
        counts[best] += 1
        running = (running * (step - 1) + preds[best]) / step
        trace.append(f"{step}:{best}({best_mae:.4f})")
    return counts / counts.sum(), trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend runs into an ensemble checkpoint.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--members", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "ensemble.json")
    parser.add_argument("--checkpoint-out", type=Path, default=ROOT / "runs" / "ensemble" / "checkpoint.pt")
    parser.add_argument("--val-len", type=int, default=336)
    parser.add_argument("--test-len", type=int, default=336)
    parser.add_argument("--rounds", type=int, default=20, help="Greedy selection steps.")
    parser.add_argument(
        "--select-on", default="val_gap", choices=["val", "val_gap"],
        help="Which validation-label evaluation selects the weights.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    missing = [str(p) for p in args.members if not p.exists()]
    if missing:
        raise SystemExit(f"Missing member checkpoint(s): {missing}")

    frame = pd.read_csv(args.train)
    length = int(frame.groupby("series_id", sort=False).size().max())
    panel, length = build_panel(frame, FeatureSpec.benchmark(), length - args.val_len - args.test_len)
    fit_slice, val_slice, test_slice = split_positions(length, args.val_len, args.test_len)

    loaded = [ForecastModel.load_checkpoint(str(p), map_location=args.device) for p in args.members]
    clip_min = min(float(m.config.get("clip_min", 0.0)) for m in loaded)
    names = [p.parent.name for p in args.members]

    evaluations = [
        ("val", val_slice, fit_slice.stop),
        ("val_gap", val_slice, fit_slice.stop - args.val_len),
        ("test", test_slice, fit_slice.stop),
    ]
    evaluations.sort(key=lambda item: item[0] != args.select_on)  # selection split first

    results: dict[str, object] = {
        "members": [str(p) for p in args.members],
        "member_names": names,
        "clip_min": clip_min,
        "selected_on": args.select_on,
        "rounds": args.rounds,
    }
    weights: np.ndarray | None = None
    for split_name, positions, origin in evaluations:
        truth, preds = member_predictions(args.members, panel, positions, origin, args.device)
        if weights is None:
            weights, trace = greedy_weights(truth, preds, clip_min, args.rounds)
            results["weights"] = weights.tolist()
            results["greedy_trace"] = trace
            picked = {n: round(float(w), 3) for n, w in zip(names, weights) if w > 0}
            print(f"selected on '{split_name}': {picked}")
        blended = np.maximum(sum(w * p for w, p in zip(weights, preds)), clip_min)
        results[split_name] = compute_metrics(truth, blended)
        results[f"{split_name}_members"] = [
            compute_metrics(truth, np.maximum(p, clip_min)) for p in preds
        ]
        print(f"[{split_name:7s}] ensemble {format_metrics(results[split_name])}")
        for name, metrics in zip(names, results[f"{split_name}_members"]):
            print(f"[{split_name:7s}]   {name:26s} {format_metrics(metrics)}")

    ensemble = ForecastModel(
        model_type="ensemble",
        members=[m.config for m in loaded],
        member_weights=weights.tolist(),
        history_len=max(m.history_len for m in loaded),
        clip_min=clip_min,
    )
    for slot, member in zip(ensemble.net.members, loaded):
        slot.load_state_dict(member.net.state_dict())
    ensemble.attach_preprocessing(loaded[0].feature_spec, loaded[0].feature_scaler)
    ensemble.attach_series_index(loaded[0].series_index)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    ensemble.save_checkpoint(str(args.checkpoint_out))

    # .to(device) is required, not decorative: load_checkpoint maps the *state dict*
    # onto the device but copies it into parameters that were constructed on CPU, so
    # the module itself stays on CPU and a CUDA batch would hit a device mismatch.
    reloaded = ForecastModel.load_checkpoint(
        str(args.checkpoint_out), map_location=args.device
    ).to(args.device)
    verify, _ = evaluate_split(
        reloaded, panel, val_slice, fit_slice.stop, args.device, reloaded.history_len,
        clip_min=reloaded.config.get("clip_min"),
    )
    results["checkpoint_val"] = verify
    print(f"[verify] reloaded ensemble {format_metrics(verify)}")
    if abs(verify["wape"] - results["val"]["wape"]) > 1e-4:
        raise SystemExit(
            f"Saved checkpoint scores {verify['wape']:.4f} on 'val' but the search reported "
            f"{results['val']['wape']:.4f}: searched and serialised blends disagree."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"wrote {args.out} and {args.checkpoint_out}")


if __name__ == "__main__":
    main()
