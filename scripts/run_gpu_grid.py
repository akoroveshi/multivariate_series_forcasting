"""Run a training grid on a single shared GPU (TU Darmstadt mlsp pool).

The CPU runs that produced the submitted models were budget-starved: PatchTST-cov
peaked at epoch 2 while seeing only ~4% of the available training windows per
epoch (window stride 24 of a possible 1). On an RTX 2080 Ti an epoch costs
seconds rather than minutes, so this script re-runs the two families with a much
smaller stride and a real epoch budget, plus a small capacity/learning-rate grid.

Concurrency is bounded because the card is shared: these models are small (0.4-0.6M
parameters, batch 256), so several fit comfortably in 11 GB, but we still cap the
number of simultaneous processes and tag each one with RTPT so other pool users can
see who is on the GPU.

Usage (on the node)::

    python scripts/run_gpu_grid.py --concurrency 4 --rtpt RB
    python scripts/run_gpu_grid.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "student" / "submission_template"

#: name -> extra CLI arguments on top of the per-model defaults below.
GRID: dict[str, tuple[str, list[str]]] = {
    # --- PatchTST-cov: does more data and a real budget close the gap? ---
    "gpu_patchtst_s6":        ("patchtst", ["--stride", "6"]),
    "gpu_patchtst_s3":        ("patchtst", ["--stride", "3"]),
    "gpu_patchtst_s6_d256":   ("patchtst", ["--stride", "6", "--d-model", "256", "--d-ff", "512"]),
    "gpu_patchtst_s6_l4":     ("patchtst", ["--stride", "6", "--n-layers", "4"]),
    "gpu_patchtst_s6_p12":    ("patchtst", ["--stride", "6", "--patch-len", "12", "--patch-stride", "6"]),
    "gpu_patchtst_s6_lr3e4":  ("patchtst", ["--stride", "6", "--lr", "3e-4"]),
    # --- LSTM+Attention ---
    "gpu_lstm_s12":           ("lstm_attention", ["--stride", "12"]),
    "gpu_lstm_s6":            ("lstm_attention", ["--stride", "6"]),
    "gpu_lstm_s12_h256":      ("lstm_attention", ["--stride", "12", "--hidden-size", "256"]),
    # --- round 2: the round-1 LSTMs were still improving at their last epoch
    # (best_epoch == epochs), so the budget, not the architecture, was binding.
    # Seeds on the leading config turn a 0.03-WAPE ranking into a real one.
    "gpu2_lstm_s6_long":      ("lstm_attention", ["--stride", "6", "--epochs", "100",
                                                  "--patience", "20", "--sgdr-t0", "100"]),
    "gpu2_lstm_s3_long":      ("lstm_attention", ["--stride", "3", "--epochs", "100",
                                                  "--patience", "20", "--sgdr-t0", "100"]),
    "gpu2_lstm_s6_h256_long": ("lstm_attention", ["--stride", "6", "--hidden-size", "256",
                                                  "--epochs", "100", "--patience", "20",
                                                  "--sgdr-t0", "100"]),
    "gpu2_lstm_s6_long_s1":   ("lstm_attention", ["--stride", "6", "--epochs", "100",
                                                  "--patience", "20", "--sgdr-t0", "100",
                                                  "--seed", "1"]),
    "gpu2_lstm_s6_long_s2":   ("lstm_attention", ["--stride", "6", "--epochs", "100",
                                                  "--patience", "20", "--sgdr-t0", "100",
                                                  "--seed", "2"]),
    # keep a strong Transformer for ensemble diversity
    "gpu2_patchtst_s3_d256":  ("patchtst", ["--stride", "3", "--d-model", "256", "--d-ff", "512",
                                            "--epochs", "60", "--patience", "15", "--sgdr-t0", "60"]),
    # --- round 3: two levers the CPU budget never let us touch. A 336-step
    # conditioning window puts a full extra week (and the lag-168 seasonal
    # pattern) inside the encoder's view; the decoder block length trades
    # rollout depth against per-block autoregression.
    "gpu3_lstm_hist336":      ("lstm_attention", ["--stride", "6", "--history-len", "336",
                                                  "--epochs", "100", "--patience", "20",
                                                  "--sgdr-t0", "100"]),
    "gpu3_lstm_hist504":      ("lstm_attention", ["--stride", "6", "--history-len", "504",
                                                  "--epochs", "100", "--patience", "20",
                                                  "--sgdr-t0", "100"]),
    "gpu3_lstm_block48":      ("lstm_attention", ["--stride", "6", "--block-len", "48",
                                                  "--epochs", "100", "--patience", "20",
                                                  "--sgdr-t0", "100"]),
    "gpu3_lstm_block12":      ("lstm_attention", ["--stride", "6", "--block-len", "12",
                                                  "--epochs", "100", "--patience", "20",
                                                  "--sgdr-t0", "100"]),
    "gpu3_lstm_l3":           ("lstm_attention", ["--stride", "6", "--encoder-layers", "3",
                                                  "--decoder-layers", "2", "--epochs", "100",
                                                  "--patience", "20", "--sgdr-t0", "100"]),
    "gpu3_patchtst_hist336":  ("patchtst", ["--stride", "6", "--history-len", "336",
                                            "--d-model", "256", "--d-ff", "512", "--epochs", "60",
                                            "--patience", "15", "--sgdr-t0", "60"]),
    # --- round 4: the Transformer's problem is the opposite of the LSTM's. Its
    # round-1 runs peaked at epoch 3-12 of 40 and then early-stopped, i.e. they
    # overfit rather than starved, so more data/epochs cannot help it -- but
    # regularisation, which we never varied, can. A strong Transformer also buys
    # the blend genuine diversity against a field of recurrent members.
    "gpu4_patchtst_do2":      ("patchtst", ["--stride", "6", "--d-model", "256", "--d-ff", "512",
                                            "--dropout", "0.2", "--epochs", "60",
                                            "--patience", "15", "--sgdr-t0", "60"]),
    "gpu4_patchtst_do3_wd":   ("patchtst", ["--stride", "6", "--d-model", "256", "--d-ff", "512",
                                            "--dropout", "0.3", "--weight-decay", "1e-3",
                                            "--epochs", "60", "--patience", "15", "--sgdr-t0", "60"]),
    "gpu4_patchtst_p8_do2":   ("patchtst", ["--stride", "6", "--d-model", "256", "--d-ff", "512",
                                            "--patch-len", "8", "--patch-stride", "4",
                                            "--dropout", "0.2", "--epochs", "60",
                                            "--patience", "15", "--sgdr-t0", "60"]),
    "gpu4_patchtst_s2_do2":   ("patchtst", ["--stride", "2", "--d-model", "256", "--d-ff", "512",
                                            "--dropout", "0.2", "--epochs", "60",
                                            "--patience", "15", "--sgdr-t0", "60"]),
    "gpu4_patchtst_huber":    ("patchtst", ["--stride", "6", "--d-model", "256", "--d-ff", "512",
                                            "--dropout", "0.2", "--loss", "huber",
                                            "--epochs", "60", "--patience", "15", "--sgdr-t0", "60"]),
    # --- full-history twins of the leading configs. Local selection must hold out
    # the last 672 steps, but the private window sits after them, so the submitted
    # members are retrained on everything with the schedule chosen locally.
    # NOTE the epoch counts: the scheduled-sampling ratio is annealed across the
    # *whole* budget, so a longer budget is not a superset of a shorter one -- it
    # spends proportionally longer at high teacher forcing. Stretching the winning
    # 30-epoch LSTM to 100 epochs made it worse (13.16 -> 13.40). The full-history
    # twins must therefore replicate the selected schedule exactly, not extend it.
    "full_lstm_s6":           ("lstm_attention", ["--stride", "6", "--epochs", "30",
                                                  "--sgdr-t0", "30", "--val-len", "0",
                                                  "--test-len", "0", "--patience", "0"]),
    "full_lstm_s12":          ("lstm_attention", ["--stride", "12", "--epochs", "30",
                                                  "--sgdr-t0", "30", "--val-len", "0",
                                                  "--test-len", "0", "--patience", "0"]),
    "full_lstm_s12_h256":     ("lstm_attention", ["--stride", "12", "--hidden-size", "256",
                                                  "--epochs", "30", "--sgdr-t0", "30",
                                                  "--val-len", "0", "--test-len", "0",
                                                  "--patience", "0"]),
    # An early-stopped run's twin must stop at the *selected* epoch while keeping
    # the original cosine period, so the replayed trajectory matches: --epochs 3
    # with --sgdr-t0 60 reproduces the first three epochs of a 60-epoch schedule,
    # whereas --epochs 60 would train 57 epochs past the point validation chose.
    "full_patchtst_s2_do2":   ("patchtst", ["--stride", "2", "--d-model", "256", "--d-ff", "512",
                                            "--dropout", "0.2", "--epochs", "3", "--sgdr-t0", "60",
                                            "--val-len", "0", "--test-len", "0", "--patience", "0"]),
    "full_patchtst_s6_d256":  ("patchtst", ["--stride", "6", "--d-model", "256", "--d-ff", "512",
                                            "--epochs", "6", "--sgdr-t0", "40", "--val-len", "0",
                                            "--test-len", "0", "--patience", "0"]),
}

#: Shared budget per family. Both drop RevIN, which the CPU ablation showed is
#: harmful at this horizon, and both use a single long cosine cycle.
BASE: dict[str, list[str]] = {
    "patchtst": [
        "--epochs", "40", "--patience", "10", "--sgdr-t0", "40", "--sgdr-tmult", "1",
        "--batch-size", "256", "--no-revin",
    ],
    "lstm_attention": [
        "--epochs", "30", "--patience", "8", "--sgdr-t0", "30", "--sgdr-tmult", "1",
        "--batch-size", "256", "--no-revin",
    ],
}


def build_jobs(names: list[str], data: Path, results: Path, runs: Path, rtpt: str) -> list[tuple[str, list[str]]]:
    jobs = []
    for name in names:
        model, extra = GRID[name]
        command = [
            sys.executable, "-u", "-m", "src.train",
            "--train", str(data / "train.csv"),
            "--model", model,
            "--device", "cuda",
            "--tag", name,
            "--checkpoint-out", str(runs / name / "checkpoint.pt"),
            "--metrics-out", str(results / f"{name}.json"),
            *BASE[model],
            *extra,
        ]
        if rtpt:
            command += ["--rtpt", rtpt]
        jobs.append((name, command))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GPU training grid.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--rtpt", default="", help="Initials for RTPT process tagging.")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    names = list(GRID) if not args.only else [n for n in args.only if n in GRID]
    unknown = set(args.only or []) - set(GRID)
    if unknown:
        raise SystemExit(f"Unknown config(s): {sorted(unknown)}")
    for d in (args.results_dir, args.runs_dir, args.logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(names, args.data_dir, args.results_dir, args.runs_dir, args.rtpt)
    print(f"{len(jobs)} configs, concurrency {args.concurrency}")
    if args.dry_run:
        for name, cmd in jobs:
            print(f"  {name}\n      {' '.join(cmd)}")
        return 0

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    pending, running, done = list(jobs), [], []
    started_at = time.time()
    while pending or running:
        while pending and len(running) < args.concurrency:
            name, cmd = pending.pop(0)
            handle = open(args.logs_dir / f"{name}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(PKG), env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((name, proc, handle, time.time()))
            print(f"[{time.strftime('%H:%M:%S')}] start {name} ({len(pending)} queued)", flush=True)
        time.sleep(5)
        for entry in list(running):
            name, proc, handle, t0 = entry
            if proc.poll() is None:
                continue
            handle.close()
            running.remove(entry)
            done.append((name, proc.returncode))
            status = "ok  " if proc.returncode == 0 else f"FAIL({proc.returncode})"
            print(f"[{time.strftime('%H:%M:%S')}] {status} {name} in {(time.time()-t0)/60:.1f} min", flush=True)

    print(f"\nfinished in {(time.time()-started_at)/60:.1f} min")
    rows = []
    for name, code in done:
        path = args.results_dir / f"{name}.json"
        if code != 0 or not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        # Full-history twins train with --val-len 0, so they have nothing to score
        # here; they are evaluated only indirectly, through the blend they join.
        if not d.get("val"):
            print(f"  {name}: full-history run, no held-out split to report")
            continue
        rows.append((d["val"]["wape"], d["test"]["wape"], d["best_epoch"], name))
    rows.sort()
    print(f"\n{'val WAPE':>9s} {'test WAPE':>10s} {'best ep':>8s}  config")
    for v, t, e, n in rows:
        print(f"{v:9.3f} {t:10.3f} {e:8d}  {n}")
    return 0 if all(c == 0 for _, c in done) else 1


if __name__ == "__main__":
    raise SystemExit(main())
