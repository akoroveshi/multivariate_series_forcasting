"""Train a forecaster on the operations benchmark (or any compatible panel).

Model selection never touches the leaderboard: the released tables contain no
validation labels, so ``train.csv`` is split chronologically into a fit slice and
two held-out windows of 336 steps each (``--val-len`` / ``--test-len``). The
first mimics the public validation split, the second mimics the private test
split, whose forecast window starts 336 steps after the last released target.

Examples
--------
Local model selection for the LSTM+Attention hybrid::

    python -m src.train --train ../../data/train.csv --model lstm_attention \\
        --checkpoint-out ../../runs/lstm/checkpoint.pt

Final model trained on the full history with the epoch budget found locally::

    python -m src.train --train ../../data/train.csv --model patchtst \\
        --val-len 0 --test-len 0 --epochs 24 \\
        --checkpoint-out ../../runs/patchtst_full/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import WindowDataset
from .evaluate import evaluate_split, evaluate_windows
from .features import FeatureScaler, FeatureSpec, SeriesPanel, split_positions
from .metrics import build_loss, format_metrics
from .model import ForecastModel, count_parameters

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "lstm_attention": {"block_len": 24, "epochs": 20, "lr": 1e-3, "batch_size": 64},
    "patchtst": {"block_len": 336, "epochs": 30, "lr": 1e-3, "batch_size": 64},
}


def set_seed(seed: int) -> None:
    """Fix Python/NumPy/Torch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scheduled_sampling_ratio(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """Linearly anneal the teacher-forcing probability across training.

    Early epochs mostly consume ground-truth previous values (fast, stable
    credit assignment); later epochs mostly consume the decoder's own
    predictions, which is the distribution seen at inference time.
    """
    if total_epochs <= 1:
        return end
    progress = min(1.0, epoch / max(1, total_epochs - 1))
    return start + (end - start) * progress


def build_panel(
    frame: pd.DataFrame, spec: FeatureSpec, fit_end: int
) -> tuple[SeriesPanel, int]:
    """Build a panel whose covariate scaler is fitted on the fit slice only."""
    work = frame.copy()
    work[spec.time_col] = pd.to_datetime(work[spec.time_col])
    work = work.sort_values([spec.series_col, spec.time_col], kind="mergesort")
    lengths = work.groupby(spec.series_col, sort=False).size().unique()
    if len(lengths) != 1:
        raise ValueError(f"All series must share one length for the positional split, got {lengths}.")
    length = int(lengths[0])

    position = work.groupby(spec.series_col, sort=False).cumcount()
    fit_rows = work.loc[(position < fit_end).to_numpy()]
    scaler = FeatureScaler.fit(fit_rows, list(spec.covariates) + list(spec.static))
    return SeriesPanel.from_frame(work, spec, scaler), length


def train_one_epoch(
    model: ForecastModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    loss_fn: torch.nn.Module,
    teacher_forcing_ratio: float,
    grad_clip: float = 1.0,
) -> float:
    """One pass over the training windows; returns the mean loss per sample."""
    model.train()
    total_loss, total_samples = 0.0, 0
    for batch in loader:
        history_target = batch["history_target"].to(device)
        history_features = batch["history_features"].to(device)
        future_features = batch["future_features"].to(device)
        static = batch["static"].to(device)
        future_target = batch["future_target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            history_target,
            history_features,
            future_features,
            static=static,
            series_index=batch["series_index"].to(device),
            future_target=future_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        horizon = future_target.size(1)
        loss = loss_fn(prediction[:, :horizon, :], future_target)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item()) * history_target.size(0)
        total_samples += history_target.size(0)
    return total_loss / max(total_samples, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multivariate forecasting model.")
    parser.add_argument("--train", required=True, type=Path, help="Training CSV with targets.")
    parser.add_argument("--checkpoint-out", required=True, type=Path)
    parser.add_argument("--metrics-out", type=Path, default=None, help="Optional JSON run log.")
    parser.add_argument("--model", default="lstm_attention", choices=sorted(MODEL_DEFAULTS))

    parser.add_argument("--history-len", type=int, default=168)
    parser.add_argument("--block-len", type=int, default=None, help="Native model output length.")
    parser.add_argument("--stride", type=int, default=12, help="Stride between training windows.")
    parser.add_argument("--val-len", type=int, default=336, help="Held-out validation steps.")
    parser.add_argument("--test-len", type=int, default=336, help="Held-out test steps (gap regime).")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", default="l1", choices=["l1", "mae", "mse", "huber", "wape"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8, help="Early-stopping patience in epochs.")
    parser.add_argument("--no-sgdr", action="store_true", help="Disable cosine warm restarts.")
    parser.add_argument("--sgdr-t0", type=int, default=5, help="Epochs until the first warm restart.")
    parser.add_argument("--sgdr-tmult", type=int, default=2)
    parser.add_argument("--teacher-forcing-start", type=float, default=0.9)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.1)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--spatial-dropout", type=float, default=0.1)
    parser.add_argument("--patch-len", type=int, default=24)
    parser.add_argument("--patch-stride", type=int, default=12)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)

    parser.add_argument("--no-revin", action="store_true", help="Ablation: disable RevIN.")
    parser.add_argument(
        "--no-series-embedding",
        action="store_true",
        help="Ablation: drop the learned per-series embedding.",
    )
    parser.add_argument("--series-embedding-dim", type=int, default=16)
    parser.add_argument(
        "--no-pointwise-head",
        action="store_true",
        help="Ablation: PatchTST without the pointwise covariate residual path.",
    )
    parser.add_argument("--pointwise-hidden", type=int, default=64)
    parser.add_argument("--no-attention", action="store_true", help="Ablation: drop attention.")
    parser.add_argument(
        "--no-covariates",
        action="store_true",
        help="Ablation: target-only model (calendar and exogenous covariates removed).",
    )
    parser.add_argument(
        "--covariate-mode",
        default="tokens",
        choices=["tokens", "none"],
        help="PatchTST: future covariates as query tokens, or vanilla flatten head.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="", help="Free-form label stored in the run log.")
    parser.add_argument(
        "--eval-protocol",
        default="rollout",
        choices=["rollout", "windows"],
        help="'rollout': one forecast origin per series (benchmark contract). "
        "'windows': rolling-origin evaluation across the whole held-out slice.",
    )
    parser.add_argument("--eval-stride", type=int, default=None, help="Origin stride for 'windows'.")
    parser.add_argument(
        "--eval-horizon", type=int, default=None, help="Forecast length per origin for 'windows'."
    )
    return parser.parse_args(argv)


def score_slice(
    model: ForecastModel,
    panel: SeriesPanel,
    positions: slice,
    history_end: int,
    args: argparse.Namespace,
    block_len: int,
    clip_min: float | None,
    device: torch.device,
) -> dict[str, float]:
    """Score a held-out slice under the configured evaluation protocol."""
    if args.eval_protocol == "windows":
        horizon = min(args.eval_horizon or block_len, positions.stop - positions.start)
        return evaluate_windows(
            model,
            panel,
            positions,
            history_len=args.history_len,
            horizon=horizon,
            stride=args.eval_stride or horizon,
            device=device,
            clip_min=clip_min,
        )
    metrics, _ = evaluate_split(
        model, panel, positions, history_end, device, args.history_len, clip_min=clip_min
    )
    return metrics


def run_experiment(panel: SeriesPanel, args: argparse.Namespace) -> dict[str, Any]:
    """Train, select on the validation slice and score the selected checkpoint.

    Works for any :class:`~src.features.SeriesPanel`, which is what lets the
    additional-dataset script reuse this exact loop.
    """
    defaults = MODEL_DEFAULTS[args.model]
    block_len = args.block_len or defaults["block_len"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    lr = args.lr or defaults["lr"]

    set_seed(args.seed)
    device = torch.device(args.device)
    length = len(next(iter(panel.series.values())))
    fit_slice, val_slice, test_slice = split_positions(length, args.val_len, args.test_len)
    if fit_slice.stop <= args.history_len + block_len:
        raise ValueError("Fit slice is too short for the requested history/block lengths.")

    train_set = WindowDataset(
        panel,
        history_len=args.history_len,
        horizon=block_len,
        stride=args.stride,
        end_position=fit_slice.stop,
    )
    if len(train_set) == 0:
        raise ValueError("No training windows were built; check history/block lengths and stride.")
    loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    clip_min = float(np.nanmin([np.nanmin(a.y[fit_slice]) for a in panel.series.values()]))
    config: dict[str, Any] = {
        "model_type": args.model,
        "num_history_features": panel.num_history_features,
        "num_future_features": panel.num_future_features,
        "num_static_features": panel.num_static_features,
        "history_len": args.history_len,
        "block_len": block_len,
        "dropout": args.dropout,
        "use_revin": not args.no_revin,
        "clip_min": clip_min,
        "num_series_slots": 1 if args.no_series_embedding else panel.num_series_slots,
        "series_embedding_dim": args.series_embedding_dim,
    }
    if args.model == "lstm_attention":
        config.update(
            hidden_size=args.hidden_size,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
            spatial_dropout=args.spatial_dropout,
            use_attention=not args.no_attention,
        )
    else:
        config.update(
            patch_len=args.patch_len,
            stride=args.patch_stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            covariate_mode="none" if args.no_covariates else args.covariate_mode,
            pointwise_head=not args.no_pointwise_head,
            pointwise_hidden=args.pointwise_hidden,
        )

    model = ForecastModel(**config).to(device)
    model.attach_preprocessing(panel.spec, panel.scaler)
    model.attach_series_index(panel.series_index)
    loss_fn = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(loader), 1)
    scheduler = (
        None
        if args.no_sgdr
        else torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, args.sgdr_t0 * steps_per_epoch), T_mult=args.sgdr_tmult
        )
    )

    print(
        f"[setup] model={args.model} params={count_parameters(model):,} "
        f"windows={len(train_set):,} steps/epoch={steps_per_epoch} "
        f"features={panel.num_history_features}/{panel.num_future_features}"
        f"+{panel.num_static_features} device={device}"
    )

    history: list[dict[str, Any]] = []
    best_score = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(epochs):
        tf_ratio = (
            scheduled_sampling_ratio(
                epoch, epochs, args.teacher_forcing_start, args.teacher_forcing_end
            )
            if args.model == "lstm_attention"
            else 0.0
        )
        train_loss = train_one_epoch(
            model, loader, optimizer, scheduler, device, loss_fn, tf_ratio, args.grad_clip
        )
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "teacher_forcing": tf_ratio,
            "lr": optimizer.param_groups[0]["lr"],
        }
        message = (
            f"epoch {epoch + 1:3d}/{epochs}  loss={train_loss:.4f}"
            f"  tf={tf_ratio:.2f}  lr={record['lr']:.2e}"
        )

        if args.val_len > 0:
            val_metrics = score_slice(
                model, panel, val_slice, fit_slice.stop, args, block_len, clip_min, device
            )
            record["val"] = val_metrics
            message += f"  val_WAPE={val_metrics['wape']:.3f}  val_MAE={val_metrics['mae']:.4f}"
            score = val_metrics["mae"]
        else:
            score = train_loss

        history.append(record)
        print(f"{message}  ({time.time() - started:.0f}s)", flush=True)

        if score < best_score - 1e-6:
            best_score, best_epoch = score, epoch + 1
            epochs_without_improvement = 0
            args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(str(args.checkpoint_out))
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"early stopping after {epoch + 1} epochs")
                break

    if best_epoch < 0:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(args.checkpoint_out))
        best_epoch = epochs

    # Final scoring of the selected checkpoint in both held-out regimes.
    best_model = ForecastModel.load_checkpoint(str(args.checkpoint_out), map_location=str(device)).to(device)
    summary: dict[str, Any] = {
        "tag": args.tag or args.model,
        "model": args.model,
        "config": {k: v for k, v in config.items()},
        "params": count_parameters(model),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_windows": len(train_set),
        "seed": args.seed,
        "runtime_seconds": round(time.time() - started, 1),
        "history": history,
    }
    if args.val_len > 0:
        summary["val"] = score_slice(
            best_model, panel, val_slice, fit_slice.stop, args, block_len, clip_min, device
        )
        print(f"[val ] {format_metrics(summary['val'])}")
    if args.test_len > 0:
        summary["test"] = score_slice(
            best_model, panel, test_slice, fit_slice.stop, args, block_len, clip_min, device
        )
        label = "gap regime" if args.eval_protocol == "rollout" else "rolling origins"
        print(f"[test] {format_metrics(summary['test'])}  ({label})")

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"checkpoint saved to {args.checkpoint_out}")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Benchmark entrypoint: read ``train.csv``, build the panel, run the experiment."""
    args = parse_args(argv)
    block_len = args.block_len or MODEL_DEFAULTS[args.model]["block_len"]

    spec = FeatureSpec.benchmark()
    if args.no_covariates:
        spec = FeatureSpec(dynamic=(), past_only=(), static=(), mask_features=())
    frame = pd.read_csv(args.train)
    length = int(frame.groupby(spec.series_col, sort=False).size().max())
    fit_end = length - args.val_len - args.test_len
    if fit_end <= args.history_len + block_len:
        raise ValueError("Fit slice is too short for the requested history/block lengths.")
    panel, _ = build_panel(frame, spec, fit_end)
    return run_experiment(panel, args)


if __name__ == "__main__":
    main()
