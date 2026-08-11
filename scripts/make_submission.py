"""Build the final submission archive.

Steps
-----
1. Retrain the members on the **full** public history (``--val-len 0
   --test-len 0``) for the epoch budget selected on the local splits, so the
   final model sees the 672 most recent steps that local model selection had to
   hold out.
2. Blend them with the weights found by ``scripts/build_ensemble.py``.
3. Attach a *context table* to the checkpoint: a tail of the public history
   (with targets) plus the public validation covariates. The private input
   directory only carries ``test_input.csv``, whose window starts 336 steps after
   the last released target, so ``predict.py`` needs that context to bridge the
   gap. Everything is inside ``checkpoint.pt`` -- no network access at inference.
4. Write validation predictions for the public leaderboard.
5. Zip ``predict.py``, ``requirements.txt``, ``checkpoint.pt`` and ``src/``.

Usage::

    python scripts/make_submission.py                 # reuse existing full-fit runs
    python scripts/make_submission.py --retrain       # retrain the members first
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "student" / "submission_template"
sys.path.insert(0, str(PKG))

from src.features import FeatureSpec  # noqa: E402
from src.model import ForecastModel  # noqa: E402

ARCHIVE_MEMBERS = ["predict.py", "requirements.txt", "checkpoint.pt"]

#: local-split run tag -> full-history twin run directory. The twin repeats the
#: architecture *and* the schedule; see run_gpu_grid.GRID for why the epoch count
#: must match rather than exceed the selected one.
TWIN_OF: dict[str, str] = {
    "gpu_lstm_s6": "full_lstm_s6",
    "gpu_lstm_s12": "full_lstm_s12",
    "gpu_lstm_s12_h256": "full_lstm_s12_h256",
    "gpu_patchtst_s6_d256": "full_patchtst_s6_d256",
    "gpu4_patchtst_s2_do2": "full_patchtst_s2_do2",
    "patchtst_norevin": "patchtst_norevin_full",
    "lstm_norevin": "lstm_norevin_full",
    "patchtst_main": "patchtst_full",
    "lstm_main": "lstm_full",
}


def build_context(
    train_frame: pd.DataFrame,
    validation_input: pd.DataFrame | None,
    spec: FeatureSpec,
    tail_steps: int,
) -> pd.DataFrame:
    """History tail (with targets) plus known future covariates (target NaN)."""
    train = train_frame.copy()
    train[spec.time_col] = pd.to_datetime(train[spec.time_col])
    train = train.sort_values([spec.series_col, spec.time_col], kind="mergesort")
    position = train.groupby(spec.series_col, sort=False).cumcount(ascending=False)
    tail = train.loc[(position < tail_steps).to_numpy()].copy()

    parts = [tail]
    if validation_input is not None:
        future = validation_input.copy()
        future[spec.time_col] = pd.to_datetime(future[spec.time_col])
        future[spec.target_col] = np.nan
        parts.append(future)

    columns = [spec.series_col, spec.time_col, *spec.covariates, *spec.static, spec.target_col]
    context = pd.concat([part.reindex(columns=columns) for part in parts], ignore_index=True)
    context = context.sort_values([spec.series_col, spec.time_col], kind="mergesort")
    for column in context.columns:
        if pd.api.types.is_float_dtype(context[column]):
            context[column] = context[column].astype("float32")
    return context.reset_index(drop=True)


def retrain_full(data_dir: Path, threads: int) -> None:
    """Retrain the CPU-stage members on the entire public history.

    Delegates to ``scripts/run_parallel.py --stage full`` rather than re-declaring
    the commands: the full-history runs must use the *same* window stride, epoch
    budget and warm-restart period as the runs that selected ``best_epoch``, and
    duplicating those constants here is how they silently drift apart. GPU-grid
    members have their own twins in ``run_gpu_grid.GRID``; run those with
    ``run_gpu_grid.py --only full_*`` and call this script without ``--retrain``.
    """
    command = [
        sys.executable, str(ROOT / "scripts" / "run_parallel.py"),
        "--data-dir", str(data_dir),
        "--threads", str(threads),
        "--stage", "full",
    ]
    print("\n--- retraining the CPU-stage members on the full public history ---", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def stage_code(out_dir: Path) -> Path:
    """Copy the inference entrypoint, its requirements and ``src/`` into the build."""
    for name in ("predict.py", "requirements.txt"):
        shutil.copy2(PKG / name, out_dir / name)
    src_out = out_dir / "src"
    if src_out.exists():
        shutil.rmtree(src_out)
    shutil.copytree(PKG / "src", src_out, ignore=shutil.ignore_patterns("__pycache__"))
    return src_out


def pack(out_dir: Path, src_out: Path) -> Path:
    archive = out_dir / "final_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ARCHIVE_MEMBERS:
            bundle.write(out_dir / name, name)
        for path in sorted(src_out.rglob("*.py")):
            bundle.write(path, str(path.relative_to(out_dir)).replace("\\", "/"))
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    return archive


def repack(out_dir: Path) -> None:
    """Refresh the archive's code without touching the trained checkpoint.

    ``predict.py`` never imports ``src.train``, so a change to the training CLI
    leaves the shipped predictions bit-for-bit identical -- but an archive whose
    ``src/`` has drifted from the repository is confusing to read. This rebuilds
    the zip around the checkpoint already on disk, so a submission that has been
    scored keeps exactly the predictions it was scored on.
    """
    checkpoint = out_dir / "checkpoint.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing {checkpoint}; nothing to repack.")
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    pack(out_dir, stage_code(out_dir))
    after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if before != after:
        raise SystemExit("repack modified checkpoint.pt; refusing to claim otherwise")
    print(f"checkpoint.pt untouched (sha256 {after[:16]}...)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final_submission.zip.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "submission")
    parser.add_argument("--ensemble-json", type=Path, default=ROOT / "results" / "ensemble.json")
    parser.add_argument("--retrain", action="store_true", help="Retrain members on the full history.")
    parser.add_argument("--threads", type=int, default=8, help="Thread budget for --retrain.")
    parser.add_argument("--context-tail", type=int, default=504, help="History steps to bundle.")
    parser.add_argument("--model-name", default="g140_patchfusion_ens", help="Leaderboard model name.")
    parser.add_argument(
        "--twins", nargs="*", default=None,
        help="Explicit full-history run directories, in blend-member order.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--repack", action="store_true",
        help="Refresh predict.py/requirements.txt/src/ inside an existing archive and "
             "exit. Leaves checkpoint.pt and the validation CSV untouched, so a "
             "submission that has already been scored keeps its predictions.",
    )
    args = parser.parse_args()

    spec = FeatureSpec.benchmark()
    if args.repack:
        repack(args.out_dir)
        return
    if args.retrain:
        retrain_full(args.data_dir, args.threads)

    # The blend weights were searched on the local split, where validation labels
    # exist. The models we actually ship are the *full-history twins* of those same
    # configurations -- same architecture and schedule, retrained on all 4,320
    # steps, because the private window sits after the 672 we had to hold out.
    if not args.ensemble_json.exists():
        raise SystemExit(f"Missing {args.ensemble_json}; run scripts/build_ensemble.py first.")
    blend = json.loads(args.ensemble_json.read_text(encoding="utf-8"))
    weights = list(blend.get("weights") or [blend["weight"], 1 - blend["weight"]])
    names = blend.get("member_names") or [Path(p).parent.name for p in blend["members"]]

    if args.twins:
        twin_dirs = [Path(t) for t in args.twins]
    else:
        twin_dirs = [ROOT / "runs" / TWIN_OF.get(n, f"{n}_full") for n in names]
    pairs = [(n, w, d) for n, w, d in zip(names, weights, twin_dirs) if w > 1e-8]
    if not pairs:
        raise SystemExit("Every blend weight is zero.")
    print("submitting blend of:")
    for name, weight, directory in pairs:
        print(f"  w={weight:5.3f}  {name:24s} -> {directory.name}")

    missing = [str(d / "checkpoint.pt") for _, _, d in pairs if not (d / "checkpoint.pt").exists()]
    if missing:
        raise SystemExit(f"Missing full-history checkpoint(s): {missing}")
    (ROOT / "results" / "selection.json").write_text(
        json.dumps({n: d.name for n, _, d in pairs}, indent=2), encoding="utf-8"
    )

    weights = [w for _, w, _ in pairs]
    members = [
        ForecastModel.load_checkpoint(str(d / "checkpoint.pt"), map_location=args.device)
        for _, _, d in pairs
    ]
    ensemble = ForecastModel(
        model_type="ensemble",
        members=[m.config for m in members],
        member_weights=weights,
        history_len=max(m.history_len for m in members),
        clip_min=min(float(m.config.get("clip_min", 0.0)) for m in members),
    )
    for slot, member in zip(ensemble.net.members, members):
        slot.load_state_dict(member.net.state_dict())
    ensemble.attach_preprocessing(members[0].feature_spec, members[0].feature_scaler)
    ensemble.attach_series_index(members[0].series_index)

    train_frame = pd.read_csv(args.data_dir / "train.csv")
    validation_input_path = args.data_dir / "validation_input.csv"
    validation_input = pd.read_csv(validation_input_path) if validation_input_path.exists() else None
    context = build_context(train_frame, validation_input, spec, args.context_tail)
    ensemble.attach_context(context)
    print(
        f"bundled context: {len(context):,} rows "
        f"({context[spec.series_col].nunique()} series, "
        f"{int(context[spec.target_col].notna().sum()):,} observed targets)"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "checkpoint.pt"
    ensemble.save_checkpoint(str(checkpoint_path))
    size_mb = checkpoint_path.stat().st_size / 1e6
    print(f"wrote {checkpoint_path} ({size_mb:.1f} MB)")

    src_out = stage_code(args.out_dir)

    # Validation predictions for the public leaderboard.
    validation_dir = args.out_dir / "_validation_input"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for name in ("validation_input.csv", "forecast_index_validation.csv"):
        shutil.copy2(args.data_dir / name, validation_dir / name)
    predictions_path = args.out_dir / f"validation_predictions_{args.model_name}.csv"
    subprocess.run(
        [
            sys.executable, "predict.py",
            "--input_dir", str(validation_dir),
            "--output_file", str(predictions_path),
            "--checkpoint", str(checkpoint_path),
        ],
        cwd=args.out_dir,
        check=True,
    )
    shutil.rmtree(validation_dir)
    predictions = pd.read_csv(predictions_path)
    index = pd.read_csv(args.data_dir / "forecast_index_validation.csv")
    assert len(predictions) == len(index), "prediction row count must match the forecast index"
    assert list(predictions.columns) == ["series_id", "timestamp", "prediction"], predictions.columns
    assert np.isfinite(predictions["prediction"]).all(), "predictions must be finite"
    print(
        f"validation predictions: {len(predictions):,} rows, "
        f"mean={predictions.prediction.mean():.3f}, min={predictions.prediction.min():.3f}, "
        f"max={predictions.prediction.max():.3f}"
    )

    pack(args.out_dir, src_out)
    print("\nnext: upload final_submission.zip and the validation CSV to the leaderboard Space")


if __name__ == "__main__":
    main()
