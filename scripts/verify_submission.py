"""End-to-end rehearsal of the private evaluation.

The private test window starts 336 steps after the last released target and the
input directory contains no observed history, so the riskiest part of the whole
submission is ``predict.py``'s gap handling -- not the network. This script
rehearses it on data we *can* score:

* ``test_input.csv``          -- covariates of the last 336 steps of ``train.csv``,
                                target column removed
* ``forecast_index_test.csv`` -- the matching ``(series_id, timestamp)`` rows
* bundled context             -- history up to 672 steps before the end (with
                                targets) plus the covariates of the intermediate
                                336 steps with the target removed

That reproduces the private layout exactly, including the unobserved gap, and the
resulting predictions are scored against the held-out targets. Then it runs the
literal command the instructors will use.

Usage::

    python scripts/verify_submission.py --checkpoint submission/checkpoint.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "student" / "submission_template"
sys.path.insert(0, str(PKG))

from src.features import FeatureSpec  # noqa: E402
from src.metrics import format_metrics  # noqa: E402
from src.model import ForecastModel  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from make_submission import build_context  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehearse the private evaluation command.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "submission" / "checkpoint.pt")
    parser.add_argument("--package", type=Path, default=ROOT / "submission")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--val-len", type=int, default=336)
    parser.add_argument("--test-len", type=int, default=336)
    parser.add_argument("--context-tail", type=int, default=504)
    args = parser.parse_args()

    from src.metrics import compute_metrics

    spec = FeatureSpec.benchmark()
    frame = pd.read_csv(args.data_dir / "train.csv")
    frame[spec.time_col] = pd.to_datetime(frame[spec.time_col])
    frame = frame.sort_values([spec.series_col, spec.time_col], kind="mergesort")
    from_end = frame.groupby(spec.series_col, sort=False).cumcount(ascending=False).to_numpy()

    fit = frame.loc[from_end >= args.val_len + args.test_len]
    middle = frame.loc[(from_end >= args.test_len) & (from_end < args.val_len + args.test_len)]
    held_out = frame.loc[from_end < args.test_len]
    print(
        f"simulated split: history={len(fit):,} rows, unobserved gap={len(middle):,} rows, "
        f"scored window={len(held_out):,} rows"
    )

    covariate_columns = [spec.series_col, spec.time_col, *spec.covariates, *spec.static]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        held_out[covariate_columns].to_csv(input_dir / "test_input.csv", index=False)
        held_out[[spec.series_col, spec.time_col]].to_csv(
            input_dir / "forecast_index_test.csv", index=False
        )

        # Re-attach a context table that stops before the gap, mirroring the
        # public release, and save it next to the real checkpoint's weights.
        model = ForecastModel.load_checkpoint(str(args.checkpoint), map_location="cpu")
        context = build_context(fit, middle[covariate_columns], spec, args.context_tail)
        model.attach_context(context)
        rehearsal_checkpoint = tmp_dir / "checkpoint.pt"
        model.save_checkpoint(str(rehearsal_checkpoint))

        output_file = tmp_dir / "predictions.csv"
        command = [
            sys.executable, "predict.py",
            "--input_dir", str(input_dir),
            "--output_file", str(output_file),
            "--checkpoint", str(rehearsal_checkpoint),
        ]
        print("\n$ " + " ".join(command[1:]))
        subprocess.run(command, cwd=args.package, check=True)

        predictions = pd.read_csv(output_file)
        assert list(predictions.columns) == ["series_id", "timestamp", "prediction"], predictions.columns
        assert len(predictions) == len(held_out), (len(predictions), len(held_out))
        assert not predictions.duplicated(["series_id", "timestamp"]).any()
        assert np.isfinite(predictions["prediction"]).all()

        merged = held_out[[spec.series_col, spec.time_col, spec.target_col]].copy()
        merged[spec.time_col] = merged[spec.time_col].astype(str)
        merged = merged.merge(predictions, on=[spec.series_col, spec.time_col], how="left", validate="one_to_one")
        assert merged["prediction"].notna().all()
        metrics = compute_metrics(merged[spec.target_col].to_numpy(), merged["prediction"].to_numpy())
        print(f"\n[rehearsal] {format_metrics(metrics)}")
        print(
            "NOTE: this score is optimistic and is NOT a performance estimate. The\n"
            "submitted checkpoint is trained on the full public history, which includes\n"
            "the window scored here. The point of the rehearsal is mechanical: that the\n"
            "required command runs, bridges the gap, and emits a well-formed CSV.\n"
            "For honest numbers see results/*.json (local split, nothing held-in)."
        )
        print("Schema, row count and finiteness checks passed; the archive is runnable as-is.")


if __name__ == "__main__":
    main()
