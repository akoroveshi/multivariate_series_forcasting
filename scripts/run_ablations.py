"""Ablation study on the benchmark's local splits.

Every configuration -- including the two reference runs -- is trained with the
same reduced budget (larger window stride, fewer epochs) so the comparisons are
apples to apples. The final models reported as our submission are trained
separately with the full budget by ``scripts/train_main.sh``.

The covariate panel is built once and reused across runs, which is why this
script calls :func:`src.train.run_experiment` directly instead of shelling out.

Usage::

    python scripts/run_ablations.py --out results/ablations.json
    python scripts/run_ablations.py --only patchtst_reference lstm_no_attention
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student" / "submission_template"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_parallel import merge_results  # noqa: E402
from src.features import FeatureSpec  # noqa: E402
from src.metrics import format_metrics  # noqa: E402
from src.train import build_panel, parse_args, run_experiment  # noqa: E402

#: ``name -> (model, needs_target_only_panel, extra CLI arguments)``
ABLATIONS: dict[str, tuple[str, bool, list[str]]] = {
    "patchtst_reference": ("patchtst", False, []),
    "patchtst_no_future_tokens": ("patchtst", False, ["--covariate-mode", "none"]),
    "patchtst_target_only": ("patchtst", True, ["--no-covariates"]),
    "patchtst_no_pointwise_head": ("patchtst", False, ["--no-pointwise-head"]),
    "patchtst_no_series_embedding": ("patchtst", False, ["--no-series-embedding"]),
    "patchtst_no_revin": ("patchtst", False, ["--no-revin"]),
    "patchtst_patch8": ("patchtst", False, ["--patch-len", "8", "--patch-stride", "4"]),
    "patchtst_patch48": ("patchtst", False, ["--patch-len", "48", "--patch-stride", "24"]),
    "patchtst_mse_loss": ("patchtst", False, ["--loss", "mse"]),
    "lstm_reference": ("lstm_attention", False, []),
    "lstm_no_attention": ("lstm_attention", False, ["--no-attention"]),
    "lstm_no_series_embedding": ("lstm_attention", False, ["--no-series-embedding"]),
    "lstm_no_revin": ("lstm_attention", False, ["--no-revin"]),
    "lstm_teacher_forcing_only": (
        "lstm_attention",
        False,
        ["--teacher-forcing-start", "1.0", "--teacher-forcing-end", "1.0"],
    ),
    "lstm_free_running_only": (
        "lstm_attention",
        False,
        ["--teacher-forcing-start", "0.0", "--teacher-forcing-end", "0.0"],
    ),
    "lstm_target_only": ("lstm_attention", True, ["--no-covariates"]),
}

#: Reduced but *identical* budget per family, so every delta is attributable to
#: the ablated component rather than to a different amount of training. The
#: absolute numbers are therefore worse than the main runs in Table 1; only the
#: within-family differences are meant to be read.
BUDGET: dict[str, list[str]] = {
    "patchtst": ["--stride", "96", "--epochs", "8", "--patience", "4", "--sgdr-t0", "8", "--sgdr-tmult", "1"],
    "lstm_attention": ["--stride", "96", "--epochs", "5", "--patience", "3", "--sgdr-t0", "5", "--sgdr-tmult", "1"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark ablation study.")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    # One file per ablation, exactly as the parallel runner writes them, and the
    # aggregate results/ablations.json is produced only by merge_results(). Two
    # writers for one aggregate file is how a stale table sneaks into the report.
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write this single run's result here (default: results/ablations/<name>.json).",
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "ablations")
    parser.add_argument("--val-len", type=int, default=336)
    parser.add_argument("--test-len", type=int, default=336)
    parser.add_argument("--history-len", type=int, default=168)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="Subset of ablation names to run.")
    args = parser.parse_args()

    selected = list(ABLATIONS) if not args.only else [name for name in args.only if name in ABLATIONS]
    unknown = set(args.only or []) - set(ABLATIONS)
    if unknown:
        raise SystemExit(f"Unknown ablation(s): {sorted(unknown)}. Available: {sorted(ABLATIONS)}")

    frame = pd.read_csv(args.train)
    length = int(frame.groupby("series_id", sort=False).size().max())
    fit_end = length - args.val_len - args.test_len

    panels: dict[bool, object] = {}
    panels[False] = build_panel(frame, FeatureSpec.benchmark(), fit_end)[0]
    if any(ABLATIONS[name][1] for name in selected):
        panels[True] = build_panel(
            frame, FeatureSpec(dynamic=(), past_only=(), static=(), mask_features=()), fit_end
        )[0]

    for name in selected:
        model, target_only, extra = ABLATIONS[name]
        argv = [
            "--train", str(args.train),
            "--checkpoint-out", str(args.runs_dir / name / "checkpoint.pt"),
            "--metrics-out", str(args.runs_dir / name / "metrics.json"),
            "--model", model,
            "--history-len", str(args.history_len),
            "--val-len", str(args.val_len),
            "--test-len", str(args.test_len),
            "--seed", str(args.seed),
            "--tag", name,
            *BUDGET[model],
            *extra,
        ]
        print(f"\n{'=' * 78}\n=== {name}  ({model})  {' '.join(extra) or 'reference'}\n{'=' * 78}", flush=True)
        started = time.time()
        summary = run_experiment(panels[target_only], parse_args(argv))
        record = {
            name: {
                "model": model,
                "extra_args": extra,
                "val": summary.get("val"),
                "test": summary.get("test"),
                "params": summary["params"],
                "best_epoch": summary["best_epoch"],
                "epochs_run": summary["epochs_run"],
                "train_windows": summary["train_windows"],
                "runtime_seconds": round(time.time() - started, 1),
            }
        }
        out = args.out or (args.results_dir / "ablations" / f"{name}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        print(f"[{name}] val {format_metrics(summary['val'])}")
        print(f"[{name}] test {format_metrics(summary['test'])}")
        print(f"[{name}] wrote {out}")

    # Single owner for the aggregate the report reads.
    merge_results(args.results_dir)


if __name__ == "__main__":
    main()
