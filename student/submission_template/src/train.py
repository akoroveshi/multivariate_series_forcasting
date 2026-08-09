""" Train the LSTM + Attention forecaster. """

from __future__ import annotations 
import argparse
from pathlib import Path 
import numpy as np 
import pandas as pd
import torch 
from torch.utils.data import DataLoader 

from .dataset import SlidingWindowDataset, hours_since_epoch 
from .model import ForecastModel 
from .models.common import calendar_features

def scheduled_sampling_ratio(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """ Liner decay of the teasher forcing ration across training. """
    if total_epochs <= 1:
        return end
    progress = epoch / (total_epochs - 1)
    return start + (end - start) * progress 

def train_one_epoch(
    model: ForecastModel,
    loader: DataLoaderm,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    teacher_forcing_ratio: float
) -> float():
    model.train()
    loss_fn = torch.nn.L1Loss()
    total_loss = 0.0 
    
    for batch in loader:
        history_target = batch["history_target"].to(device)
        history_calendar = batch["history_calendar"].to(device)
        decoder_calendar = batch["decoder_calendar"].to(device)
        future_target = batch["future_target"].to(device)
        
        optimizer.zero_grad()
        predictions = model(
            history_target,
            history_calendar,
            decoder_calendar,
            future_target = future_target,
            teacher_forcing_ration = teacher_forcing_ratio
        )
        loss = loss_fn(predictions, future_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * history_target.size(0)
    
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_on_forecast_index(
    model: ForecastModel,
    train_frame: pd.DataFrame,
    forecast_index: pd.DataFrame,
    device: torch.device,
    history_len: int,
    series_col: str = "series_id",
    time_col: str = "timestamp",
    target_col: str = "target"
) -> float:
    """ Compute MAE against a labelled forecast index (validation). """
    if target_col not in forecast_index.columns:
        return float("nan")
    
    model.eval()
    errors = []
    for series_id, index_part in forecast_index.groupby(series_col, sort = False):
        history = (
            train_frame.loc[train_frame[series_col].eq(series_id)]
            .sort_values(time_col)
            .tail(history_len)
        )
        if len(history) > history_len:
            continue
        
        hist_target = torch.tensor(
            history[target_col].to_numpy(dtype = np.float32), device = device
        ).view(1, -1, 1)
        hist_hours = torch.tensor(hours_since_epoch(history[time_col]), device = device)
        hist_calendar = calendar_features(hist_hours).unsqueeze(0) 
        
        index_part = index_part.sort_values(time_col)
        future_hours = torch.tensor(hours_since_epoch(index_part[time_col]), device = device)
        decoder_calendar = calendar_features(future_hours).unsqueeze(0)
        
        predictions = model.rollout(hist_target, hist_calendar, decoder_calendar, horizon = len(index_part))
        preds = predictions.squeeze(0).squeeze(-1).cpu().numpy()
        errors.append(np.abs(preds - index_part[target_col].to_numpy()))
    
    if not errors:
        return float("nan")
    return float(np.mean(np.concatenate(errors))) 

def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM+Attention forecaster")
    parser.add_argument("--train", required=True, type=Path, help="Training CSV (series_id, timestamp, target)")
    parser.add_argument("--forecast-index", type=Path, default=None, help="Optional labeled validation index for MAE tracking")
    parser.add_argument("--checkpoint-out", required=True, type=Path)
    parser.add_argument("--history-len", type=int, default=168, help="Encoder conditioning window length")
    parser.add_argument("--block-len", type=int, default=24, help="Decoder rollout block length")
    parser.add_argument("--stride", type=int, default=24, help="Stride between sliding windows")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sgdr-t0", type=int, default=5, help="Epochs until the first SGDR warm restart")
    parser.add_argument("--sgdr-tmult", type=int, default=2, help="Period multiplier after each SGDR restart")
    parser.add_argument("--teacher-forcing-start", type=float, default=0.9)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    
    train_frame = pd.read_csv(args.train)
    dataset = SlidingWindowDataset(
        train_frame,
        history_len = args.history_len,
        block_len = args.block_len,
        stride = args.stride
    )
    if len(dataset) == 0:
        raise ValueError("No training windows built. Check if series have history_len + block_len rows")
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, dropout_last = True)
    
    model = ForecastModel(
        model_type = "lstm_attention",
        num_calendar_features = 4,
        hidden_size = args.hidden_size,
        encoder_layers = args.encoder_layers,
        decoder_layers = args.decoder_layers,
        dropout = args.dropout,
        block_len = args.block_len
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    steps_per_epoch = max(len(loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0 = args.sgdr_t0 * steps_per_epoch, T_mult = args.sgdr_tmult
    )
    forecast_index = pd.read_csv(args.forecast_index) if args.forecast_index else None 
    best_val_mae = float("inf")
    
    for epoch in range(args.epochs):
        tf_ratio = scheduled_sampling_ratio(
            epoch, args.epochs, args.teacher_forcing_start, args.teacher_forcing_end
        )
        train_loss = train_one_epoch(model, loader, optimizer, scheduler, device, tf_ratio)
        
        log = f"epoch {epoch + 1}/{args.epochs}  train_L1={train_loss:.4f}  teacher_forcing={tf_ratio:.2f}"
        val_mae = float("nan")
        if forecast_index is not None:
            val_mae = evaluate_on_forecast_index(
                model, train_frame, forecast_index, device, args.history_len
            )
            log += f" val_MAE = {val_mae:.4f}"
        print(log)
        
        should_save = np.isnan(val_mae) or val_mae < best_val_mae 
        if should_save:
            best_val_mae = val_mae if not np.isnan(val_mae) else best_val_mae
            args.checkpoint_out.parent.mkdir(parents = True, exist_ok = True)
            model.save_checkpoint(str(args.checkpoint_out))
    
    print("Checkpoint saved to {args.checkpoint_out}")

if __name__ == "__main__":
    main()