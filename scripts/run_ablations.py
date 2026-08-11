"""Ablation study on the benchmark's local splits.

Every configuration -- including the two reference runs -- is trained with the
same budget, so each delta is attributable to the ablated component rather than
to a different amount of training.

Two things about the design are deliberate.

*The reference is the shipped configuration.* Both references are RevIN-free at
the stride and epoch budget of the blend members, so every row answers "what
happens if I change this one thing about the model we actually submit". RevIN
therefore appears as an \\+RevIN* row with a positive delta rather than as a
removal with a negative one.

*The budget matters, so it is selectable.* ``--budget full`` reproduces the
members' own training budget and is what the report uses. ``--budget cpu`` is the
starved variant that fits on a laptop; it ranks each row against its own
reference correctly but, as the architecture comparison in the paper shows,
starvation can flip a conclusion. Do not mix rows from the two.

The covariate panel is built once and reused across runs, which is why this
script calls :func:`src.train.run_experiment` directly instead of shelling out.

Usage::

    python scripts/run_ablations.py --budget full --device cuda --rtpt RB
    python scripts/run_ablations.py --budget cpu --only patchtst_reference
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
    # The references are RevIN-free, so RevIN is *added* here, not removed.
    "patchtst_revin": ("patchtst", False, ["--revin"]),
    "patchtst_patch8": ("patchtst", False, ["--patch-len", "8", "--patch-stride", "4"]),
    "patchtst_patch48": ("patchtst", False, ["--patch-len", "48", "--patch-stride", "24"]),
    "patchtst_mse_loss": ("patchtst", False, ["--loss", "mse"]),
    "lstm_reference": ("lstm_attention", False, []),
    "lstm_no_attention": ("lstm_attention", False, ["--no-attention"]),
    "lstm_no_series_embedding": ("lstm_attention", False, ["--no-series-embedding"]),
    "lstm_revin": ("lstm_attention", False, ["--revin"]),
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

#: ``budget -> model -> arguments``. Identical within a budget across every row.
#:
#: ``full`` is the budget the submitted blend members were trained at (see
#: ``run_gpu_grid.BASE``): the reference rows here are the same configurations as
#: ``gpu_patchtst_s6`` and ``gpu_lstm_s6``, which is a useful cross-check --- they
#: should reproduce those runs' numbers to within the ~0.07 WAPE noise floor.
#: ``cpu`` is the starved variant; it needs no GPU but its absolute numbers are
#: several WAPE points worse and its rankings are less trustworthy.
BUDGETS: dict[str, dict[str, list[str]]] = {
    "full": {
        "patchtst": [
            "--stride", "6", "--epochs", "40", "--patience", "10",
            "--sgdr-t0", "40", "--sgdr-tmult", "1", "--batch-size", "256", "--no-revin",
        ],
        "lstm_attention": [
            "--stride", "6", "--epochs", "30", "--patience", "8",
            "--sgdr-t0", "30", "--sgdr-tmult", "1", "--batch-size", "256", "--no-revin",
        ],
    },
    "cpu": {
        "patchtst": [
            "--stride", "96", "--epochs", "8", "--patience", "4",
            "--sgdr-t0", "8", "--sgdr-tmult", "1", "--no-revin",
        ],
        "lstm_attention": [
            "--stride", "96", "--epochs", "5", "--patience", "3",
            "--sgdr-t0", "5", "--sgdr-tmult", "1", "--no-revin",
        ],
    },
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
    parser.add_argument(
        "--budget", choices=sorted(BUDGETS), default="full",
        help="'full' reproduces the blend members' training budget (needs a GPU); "
             "'cpu' is the starved variant.",
    )
    # Per-row overrides on top of the chosen budget. Identical across rows either way.
    parser.add_argument("--stride", type=int, default=None, help="Override window stride.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch budget.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="e.g. cuda")
    parser.add_argument("--rtpt", default="", help="Initials for RTPT tagging.")
    parser.add_argument("--suffix", default="", help="Appended to each result name.")
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip the aggregate rebuild. Use when several processes each run a "
             "--only subset: whichever finishes last would otherwise publish an "
             "aggregate built from however many files happened to exist by then.",
    )
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
        record_name = f"{name}{args.suffix}"
        argv = [
            "--train", str(args.train),
            "--checkpoint-out", str(args.runs_dir / f"{name}{args.suffix}" / "checkpoint.pt"),
            "--metrics-out", str(args.runs_dir / f"{name}{args.suffix}" / "metrics.json"),
            "--model", model,
            "--history-len", str(args.history_len),
            "--val-len", str(args.val_len),
            "--test-len", str(args.test_len),
            "--seed", str(args.seed),
            "--tag", record_name,
            # Budget first, ablation second: an ablation that flips a boolean the
            # budget already set (RevIN) has to be able to win, and argparse keeps
            # the last occurrence.
            *BUDGETS[args.budget][model],
            *extra,
        ]
        if args.stride is not None:
            argv += ["--stride", str(args.stride)]
        if args.epochs is not None:
            argv += ["--epochs", str(args.epochs), "--sgdr-t0", str(args.epochs)]
        if args.batch_size is not None:
            argv += ["--batch-size", str(args.batch_size)]
        if args.device:
            argv += ["--device", args.device]
        if args.rtpt:
            argv += ["--rtpt", args.rtpt]
        print(f"\n{'=' * 78}\n=== {name}  ({model})  {' '.join(extra) or 'reference'}\n{'=' * 78}", flush=True)
        started = time.time()
        summary = run_experiment(panels[target_only], parse_args(argv))
        record_name = f"{name}{args.suffix}"
        record = {
            record_name: {
                "model": model,
                "budget": args.budget,
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
        out = args.out or (args.results_dir / f"ablations{args.suffix}" / f"{record_name}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        print(f"[{name}] val {format_metrics(summary['val'])}")
        print(f"[{name}] test {format_metrics(summary['test'])}")
        print(f"[{name}] wrote {out}")

    # Single owner for the aggregate the report reads.
    if args.no_merge:
        print("skipping merge (--no-merge); run merge_results when every subset is done")
    elif not args.suffix:
        merge_results(args.results_dir)
    else:
        merged = {}
        for path in sorted((args.results_dir / f"ablations{args.suffix}").glob("*.json")):
            merged.update(json.loads(path.read_text(encoding="utf-8")))
        target = args.results_dir / f"ablations{args.suffix}.json"
        target.write_text(json.dumps(merged, indent=2, default=float), encoding="utf-8")
        print(f"merged {len(merged)} ablations -> {target}")


if __name__ == "__main__":
    main()
