"""Score the course reference baselines on the local held-out splits.

The public release has no validation labels, so all reported numbers come from a
chronological split of ``train.csv``:

* ``local_val``  -- last 672..337 steps: forecast starts right after the history
  (public validation regime).
* ``local_test`` -- last 336 steps: forecast starts 336 steps after the history
  (private test regime, the model must bridge an unobserved gap).

Usage::

    python scripts/run_local_baselines.py --train data/train.csv --out results/baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student" / "submission_template"))

from src.baselines import make_all_baselines  # noqa: E402
from src.metrics import compute_metrics, format_metrics  # noqa: E402

SERIES_COL, TIME_COL, TARGET_COL = "series_id", "timestamp", "target"


def chronological_parts(frame: pd.DataFrame, val_len: int, test_len: int) -> dict[str, pd.DataFrame]:
    """Split a long table into fit / local_val / local_test by per-series position."""
    work = frame.copy()
    work[TIME_COL] = pd.to_datetime(work[TIME_COL])
    work = work.sort_values([SERIES_COL, TIME_COL], kind="mergesort")
    position = work.groupby(SERIES_COL, sort=False).cumcount().to_numpy()
    length = int(work.groupby(SERIES_COL, sort=False).size().max())
    fit_end = length - val_len - test_len
    return {
        "fit": work.loc[position < fit_end].reset_index(drop=True),
        "local_val": work.loc[(position >= fit_end) & (position < fit_end + val_len)].reset_index(drop=True),
        "local_test": work.loc[position >= fit_end + val_len].reset_index(drop=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score reference baselines on the local splits.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "baselines_local.json")
    parser.add_argument("--val-len", type=int, default=336)
    parser.add_argument("--test-len", type=int, default=336)
    args = parser.parse_args()

    frame = pd.read_csv(args.train)
    parts = chronological_parts(frame, args.val_len, args.test_len)
    fit = parts["fit"]
    print(
        f"fit rows={len(fit):,}  local_val rows={len(parts['local_val']):,}  "
        f"local_test rows={len(parts['local_test']):,}"
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    for split in ("local_val", "local_test"):
        labels = parts[split]
        index = labels[[SERIES_COL, TIME_COL]]
        # The gap regime is handled implicitly: baselines only ever see ``fit``,
        # so for local_test they extrapolate 336 steps further than for local_val.
        for name, predictions in make_all_baselines(fit, index).items():
            merged = labels[[SERIES_COL, TIME_COL, TARGET_COL]].merge(
                predictions, on=[SERIES_COL, TIME_COL], how="left", validate="one_to_one"
            )
            metrics = compute_metrics(merged[TARGET_COL].to_numpy(), merged["prediction"].to_numpy())
            results.setdefault(name, {})[split] = metrics
            print(f"{split:11s} {name:18s} {format_metrics(metrics)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
