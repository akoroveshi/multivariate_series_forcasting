"""Run every experiment concurrently under a CPU thread budget.

Twenty-two training runs (2 main models, 16 ablations, 4 additional-dataset runs)
are independent, so running them one at a time wastes most of the machine: a
small Transformer or LSTM scales badly past a couple of CPU threads, and a single
6-thread job leaves far more throughput on the table than six 1-thread jobs.

This runner therefore keeps a *thread budget* rather than a worker count. Each job
declares how many threads it wants (the two long main runs take several, the short
ablations take one) and the scheduler starts jobs while the running total fits in
``--threads``. Big jobs are queued first so they start immediately and the short
ones backfill as capacity frees up.

Per-job results are written to separate JSON files and merged at the end, so
concurrent jobs never race on a shared output file.

Usage::

    python scripts/run_parallel.py                    # everything
    python scripts/run_parallel.py --stage main       # only the main models
    python scripts/run_parallel.py --stage ablations --threads 6
    python scripts/run_parallel.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "student" / "submission_template"


@dataclass
class Job:
    """One training run: a command, a thread request and a log file."""

    name: str
    stage: str
    command: list[str]
    threads: int = 1
    cwd: Path = ROOT
    process: subprocess.Popen | None = field(default=None, repr=False)
    started: float = 0.0
    finished: float = 0.0
    returncode: int | None = None

    @property
    def duration(self) -> float:
        end = self.finished or time.time()
        return end - self.started if self.started else 0.0


def train_command(**flags: object) -> list[str]:
    """Build a ``python -m src.train`` invocation from keyword flags."""
    command = [sys.executable, "-m", "src.train"]
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            command.append(flag)
        elif value is not False and value is not None:
            command += [flag, str(value)]
    return command


def main_jobs(data: Path, epochs_patchtst: int, epochs_lstm: int) -> list[Job]:
    """The two models we actually submit, trained on the local split."""
    return [
        Job(
            name="patchtst_main",
            stage="main",
            threads=4,
            cwd=PKG,
            command=train_command(
                train=data / "train.csv",
                model="patchtst",
                stride=24,
                epochs=epochs_patchtst,
                sgdr_t0=6,
                sgdr_tmult=1,
                patience=6,
                tag="patchtst_main",
                checkpoint_out=ROOT / "runs" / "patchtst_main" / "checkpoint.pt",
                metrics_out=ROOT / "results" / "patchtst_main.json",
            ),
        ),
        Job(
            name="lstm_main",
            stage="main",
            threads=4,
            cwd=PKG,
            command=train_command(
                train=data / "train.csv",
                model="lstm_attention",
                stride=48,
                epochs=epochs_lstm,
                sgdr_t0=4,
                sgdr_tmult=1,
                patience=4,
                tag="lstm_main",
                checkpoint_out=ROOT / "runs" / "lstm_main" / "checkpoint.pt",
                metrics_out=ROOT / "results" / "lstm_main.json",
            ),
        ),
    ]


def full_history_jobs(data: Path, results_dir: Path, epochs_patchtst: int, epochs_lstm: int) -> list[Job]:
    """Retrain both members on the entire public history, for the submission.

    Local model selection has to hold out the last 672 steps, but the private test
    window sits *after* them, so the submitted model should see them. The epoch
    budget is the one selected locally (read from the run logs when available).
    """
    budgets = {"patchtst": epochs_patchtst, "lstm_attention": epochs_lstm}
    for model, log in (("patchtst", "patchtst_main.json"), ("lstm_attention", "lstm_main.json")):
        path = results_dir / log
        if path.exists():
            budgets[model] = int(json.loads(path.read_text(encoding="utf-8"))["best_epoch"])
    print(f"full-history epoch budgets: {budgets}")

    return [
        Job(
            name="patchtst_full",
            stage="full",
            threads=4,
            cwd=PKG,
            command=train_command(
                train=data / "train.csv",
                model="patchtst",
                stride=24,
                epochs=budgets["patchtst"],
                sgdr_t0=6,
                sgdr_tmult=1,
                val_len=0,
                test_len=0,
                patience=0,
                tag="patchtst_full",
                checkpoint_out=ROOT / "runs" / "patchtst_full" / "checkpoint.pt",
                metrics_out=ROOT / "runs" / "patchtst_full" / "metrics.json",
            ),
        ),
        Job(
            name="lstm_full",
            stage="full",
            threads=4,
            cwd=PKG,
            command=train_command(
                train=data / "train.csv",
                model="lstm_attention",
                stride=48,
                epochs=budgets["lstm_attention"],
                sgdr_t0=4,
                sgdr_tmult=1,
                val_len=0,
                test_len=0,
                patience=0,
                tag="lstm_full",
                checkpoint_out=ROOT / "runs" / "lstm_full" / "checkpoint.pt",
                metrics_out=ROOT / "runs" / "lstm_full" / "metrics.json",
            ),
        ),
    ]


def variant_jobs(data: Path, epochs_patchtst: int, epochs_lstm: int) -> list[Job]:
    """Re-run both members without RevIN, at the main and full-history budgets.

    The ablation study found that reversible instance normalisation *hurts* on this
    benchmark, on validation as well as in the gap regime. That is a design signal,
    not just a table row, so we re-train the candidates and let the validation split
    decide which variant ships. Four jobs at two threads each so they all run at
    once.
    """
    jobs = []
    for model, stride, sgdr, epochs in (
        ("patchtst", 24, 6, epochs_patchtst),
        ("lstm_attention", 48, 4, epochs_lstm),
    ):
        short = "patchtst" if model == "patchtst" else "lstm"
        for suffix, extra in (
            ("norevin", {"val_len": 336, "test_len": 336, "patience": 6,
                         "metrics_out": ROOT / "results" / f"{short}_norevin.json"}),
            ("norevin_full", {"val_len": 0, "test_len": 0, "patience": 0,
                              "metrics_out": ROOT / "runs" / f"{short}_norevin_full" / "metrics.json"}),
        ):
            name = f"{short}_{suffix}"
            jobs.append(
                Job(
                    name=name,
                    stage="variants",
                    threads=2,
                    cwd=PKG,
                    command=train_command(
                        train=data / "train.csv",
                        model=model,
                        stride=stride,
                        epochs=epochs,
                        sgdr_t0=sgdr,
                        sgdr_tmult=1,
                        no_revin=True,
                        tag=name,
                        checkpoint_out=ROOT / "runs" / name / "checkpoint.pt",
                        **extra,
                    ),
                )
            )
    return jobs


def ablation_jobs(data: Path) -> list[Job]:
    """One process per ablation, each writing its own result file."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_ablations import ABLATIONS  # noqa: E402  (local import keeps the list in one place)

    out_dir = ROOT / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, (model, _, _) in ABLATIONS.items():
        jobs.append(
            Job(
                name=f"ablation_{name}",
                stage="ablations",
                # Sequential decoders and the 4x token count of patch length 8
                # starve on a single thread.
                threads=2 if (model == "lstm_attention" or name == "patchtst_patch8") else 1,
                command=[
                    sys.executable,
                    str(ROOT / "scripts" / "run_ablations.py"),
                    "--train", str(data / "train.csv"),
                    "--out", str(out_dir / f"{name}.json"),
                    "--only", name,
                ],
            )
        )
    return jobs


def jena_jobs(data: Path, epochs: int) -> list[Job]:
    """One process per (target, model) pair on the additional dataset."""
    out_dir = ROOT / "results" / "jena"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for target in ("temperature", "pressure"):
        for model in ("patchtst", "lstm_attention"):
            jobs.append(
                Job(
                    name=f"jena_{target}_{model}",
                    stage="jena",
                    threads=2 if model == "lstm_attention" else 1,
                    command=[
                        sys.executable,
                        str(ROOT / "scripts" / "jena_experiment.py"),
                        "--csv", str(data / "jena_climate_2009_2016.csv"),
                        "--out", str(out_dir / f"{target}_{model}.json"),
                        "--targets", target,
                        "--models", model,
                        "--epochs", str(epochs),
                    ],
                )
            )
    return jobs


def merge_results(results_dir: Path) -> None:
    """Fold the per-job files into the two aggregate JSONs the report reads."""
    ablations_dir = results_dir / "ablations"
    if ablations_dir.exists():
        merged: dict[str, object] = {}
        paths = sorted(ablations_dir.glob("*.json"))
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            clashes = set(payload) & set(merged)
            if clashes:
                raise SystemExit(f"{path.name} redefines already-merged ablation(s): {clashes}")
            merged.update(payload)
        # One file per ablation, so a count mismatch means a result was dropped --
        # which is easy to miss when the output is a table with one fewer row.
        if len(merged) != len(paths):
            raise SystemExit(
                f"merged {len(merged)} ablations from {len(paths)} files; something was lost."
            )
        if merged:
            (results_dir / "ablations.json").write_text(
                json.dumps(merged, indent=2, default=float), encoding="utf-8"
            )
            print(f"merged {len(merged)} ablations -> results/ablations.json")

    jena_dir = results_dir / "jena"
    if jena_dir.exists():
        merged_jena: dict[str, dict] = {}
        for path in sorted(jena_dir.glob("*.json")):
            for target, payload in json.loads(path.read_text(encoding="utf-8")).items():
                entry = merged_jena.setdefault(target, {})
                for key, value in payload.items():
                    if key == "baselines":
                        entry.setdefault("baselines", {}).update(value)
                    else:
                        entry[key] = value
        if merged_jena:
            (results_dir / "jena.json").write_text(
                json.dumps(merged_jena, indent=2, default=float), encoding="utf-8"
            )
            print(f"merged {len(merged_jena)} Jena targets -> results/jena.json")


def run(jobs: list[Job], total_threads: int, logs_dir: Path) -> int:
    """Thread-budget scheduler: start jobs while the running total fits."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = sorted(jobs, key=lambda job: -job.threads)
    running: list[Job] = []
    done: list[Job] = []
    started_at = time.time()

    while pending or running:
        used = sum(job.threads for job in running)
        launched = True
        while launched:
            launched = False
            for job in list(pending):
                if used + job.threads <= total_threads or not running:
                    env = dict(os.environ)
                    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                        env[variable] = str(job.threads)
                    env["PYTHONUNBUFFERED"] = "1"
                    log_path = logs_dir / f"{job.name}.log"
                    job.log_handle = open(log_path, "w", encoding="utf-8")  # type: ignore[attr-defined]
                    job.process = subprocess.Popen(
                        job.command,
                        cwd=str(job.cwd),
                        env=env,
                        stdout=job.log_handle,  # type: ignore[attr-defined]
                        stderr=subprocess.STDOUT,
                    )
                    job.started = time.time()
                    pending.remove(job)
                    running.append(job)
                    used += job.threads
                    launched = True
                    print(
                        f"[{time.strftime('%H:%M:%S')}] start  {job.name} "
                        f"({job.threads}t, {used}/{total_threads} in use, "
                        f"{len(pending)} queued)",
                        flush=True,
                    )
                    break

        time.sleep(5)
        for job in list(running):
            code = job.process.poll()  # type: ignore[union-attr]
            if code is None:
                continue
            job.finished = time.time()
            job.returncode = code
            job.log_handle.close()  # type: ignore[attr-defined]
            running.remove(job)
            done.append(job)
            status = "ok  " if code == 0 else f"FAIL({code})"
            print(
                f"[{time.strftime('%H:%M:%S')}] {status} {job.name} "
                f"in {job.duration / 60:.1f} min  ({len(pending)} queued, {len(running)} running)",
                flush=True,
            )

    failures = [job for job in done if job.returncode != 0]
    print(f"\n{len(done) - len(failures)}/{len(done)} jobs succeeded "
          f"in {(time.time() - started_at) / 60:.1f} min wall clock")
    for job in failures:
        print(f"  FAILED {job.name} -> logs/{job.name}.log")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all experiments in parallel.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--logs-dir", type=Path, default=ROOT / "logs")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(2, (os.cpu_count() or 4)),
        help="Total CPU threads to distribute across concurrent jobs.",
    )
    parser.add_argument(
        "--stage", nargs="*", default=["main", "ablations", "jena"],
        choices=["main", "ablations", "jena", "full", "variants"],
        help="'full' retrains the submission members on the entire public history.",
    )
    parser.add_argument("--epochs-patchtst", type=int, default=12)
    parser.add_argument("--epochs-lstm", type=int, default=8)
    parser.add_argument("--epochs-jena", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs: list[Job] = []
    if "main" in args.stage:
        jobs += main_jobs(args.data_dir, args.epochs_patchtst, args.epochs_lstm)
    if "full" in args.stage:
        jobs += full_history_jobs(
            args.data_dir, args.results_dir, args.epochs_patchtst, args.epochs_lstm
        )
    if "ablations" in args.stage:
        jobs += ablation_jobs(args.data_dir)
    if "jena" in args.stage:
        jobs += jena_jobs(args.data_dir, args.epochs_jena)
    if "variants" in args.stage:
        jobs += variant_jobs(args.data_dir, args.epochs_patchtst, args.epochs_lstm)

    print(f"{len(jobs)} jobs, thread budget {args.threads}")
    if args.dry_run:
        for job in jobs:
            print(f"  {job.threads}t  {job.name}\n      {' '.join(map(str, job.command))}")
        return 0

    status = run(jobs, args.threads, args.logs_dir)
    merge_results(args.results_dir)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
