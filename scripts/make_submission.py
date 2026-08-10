"""Build the final submission archive.

Steps
-----
1. Retrain the members on the **full** public history (``--val-len 0
   --test-len 0``) for the epoch budget selected on the local splits, so the
   final model sees the 672 most recent steps that local model selection had to
   hold out.
2. Blend them with the weight found by ``scripts/build_ensemble.py``.
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
    """Retrain both members on the entire public history.

    Delegates to ``scripts/run_parallel.py --stage full`` rather than re-declaring
    the commands: the full-history runs must use the *same* window stride and
    warm-restart period as the runs that selected ``best_epoch``, and duplicating
    those constants here is how they silently drift apart.
    """
    command = [
        sys.executable, str(ROOT / "scripts" / "run_parallel.py"),
        "--data-dir", str(data_dir),
        "--threads", str(threads),
        "--stage", "full",
    ]
    print("\n--- retraining both members on the full public history ---", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final_submission.zip.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "submission")
    parser.add_argument("--ensemble-json", type=Path, default=ROOT / "results" / "ensemble.json")
    parser.add_argument("--retrain", action="store_true", help="Retrain members on the full history.")
    parser.add_argument("--threads", type=int, default=8, help="Thread budget for --retrain.")
    parser.add_argument("--context-tail", type=int, default=504, help="History steps to bundle.")
    parser.add_argument("--model-name", default="g140_patchfusion_ens", help="Leaderboard model name.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    spec = FeatureSpec.benchmark()
    if args.retrain:
        retrain_full(args.data_dir, args.threads)

    # Per family, ship whichever variant wins on the *validation* split. The RevIN
    # ablation was strong enough to be worth re-running at full budget, so both
    # candidates exist; the test split plays no part in this choice.
    families = {
        "patchtst": ["patchtst_main", "patchtst_norevin"],
        "lstm_attention": ["lstm_main", "lstm_norevin"],
    }
    chosen: dict[str, str] = {}
    for family, candidates in families.items():
        scored = []
        for tag in candidates:
            log = ROOT / "results" / f"{tag}.json"
            if log.exists():
                scored.append((json.loads(log.read_text(encoding="utf-8")), tag))
        if not scored:
            raise SystemExit(f"No run logs for {family}; run scripts/run_all.sh first.")
        summary, tag = min(scored, key=lambda item: item[0]["val"]["mae"])
        chosen[family] = tag
        losers = ", ".join(
            f"{other}={s['val']['mae']:.4f}" for s, other in scored if other != tag
        )
        print(
            f"{family}: selected {tag} (val MAE {summary['val']['mae']:.4f}, "
            f"best epoch {summary['best_epoch']})" + (f"; rejected {losers}" if losers else "")
        )

    full_dirs = {family: ROOT / "runs" / f"{tag}_full" for family, tag in chosen.items()}
    missing = [str(d / "checkpoint.pt") for d in full_dirs.values() if not (d / "checkpoint.pt").exists()]
    if missing:
        raise SystemExit(f"Missing full-history checkpoints: {missing}. Re-run with --retrain.")
    (ROOT / "results" / "selection.json").write_text(
        json.dumps(chosen, indent=2), encoding="utf-8"
    )

    if not args.ensemble_json.exists():
        raise SystemExit(f"Missing {args.ensemble_json}; run scripts/build_ensemble.py first.")
    weight = float(json.loads(args.ensemble_json.read_text(encoding="utf-8"))["weight"])
    print(f"blending weight (PatchTST share): {weight:.2f}")

    members = [
        ForecastModel.load_checkpoint(
            str(full_dirs["patchtst"] / "checkpoint.pt"), map_location=args.device
        ),
        ForecastModel.load_checkpoint(
            str(full_dirs["lstm_attention"] / "checkpoint.pt"), map_location=args.device
        ),
    ]
    ensemble = ForecastModel(
        model_type="ensemble",
        members=[m.config for m in members],
        member_weights=[weight, 1.0 - weight],
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

    for name in ("predict.py", "requirements.txt"):
        shutil.copy2(PKG / name, args.out_dir / name)
    src_out = args.out_dir / "src"
    if src_out.exists():
        shutil.rmtree(src_out)
    shutil.copytree(PKG / "src", src_out, ignore=shutil.ignore_patterns("__pycache__"))

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

    archive = args.out_dir / "final_submission.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in ARCHIVE_MEMBERS:
            bundle.write(args.out_dir / name, name)
        for path in sorted(src_out.rglob("*.py")):
            bundle.write(path, str(path.relative_to(args.out_dir)).replace("\\", "/"))
    print(f"wrote {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    print("\nnext: upload final_submission.zip and the validation CSV to the leaderboard Space")


if __name__ == "__main__":
    main()
