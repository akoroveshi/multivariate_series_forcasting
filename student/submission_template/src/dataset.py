""" Sliding-window dataset construction for LSTM + Attention.
Builds widows from a long table (series_id, timestap, target)
also used by baselines.py.
Calendar features (cyclical hour-of-day / day-of-week) are derived from
the timestamp and used as known feature inputs that the decoder conditions 
on without leaking the target.
"""

from __future__ import annotations
import numpy as np 
import pandas as pd
import torch 
from torch.utils.data import Dataset

from .models.common import calendar_features 

def hours_since_epoch(timestamps: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(timestamps)
    return (parsed.astype("int64") // 10**9 // 3600).to_numpy()

class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        train_frame: pd.DataFrame,
        history_len: int = 168,
        block_len: int = 24,
        stride: int = 24,
        series_col: str = "series_id",
        time_col: str = "timestamp",
        target_col: str = "target",
    ) -> None:
        
        self.history_len = history_len 
        self.block_len = block_len 
        self._samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        
        for _, group in train_frame.sort_values(time_col).groupby(series_col, sort = False):
            target = group[target_col].to_numpy(dtype = np.float32)
            hours = hours_since_epoch(group[time_col]) 
            window_len = history_len + block_len 
            
            if len(target) < window_len:
                continue
            
            for start in range(0, len(target) - window_len + 1, stride):
                hist_target = target[start: start + history_len]
                hist_hours = hours[start: start + history_len]
                
                future_start = start + history_len 
                future_target = target[future_start: future_start + block_len]
                future_hours = hours[future_start: future_start + block_len]
                
                self._samples.append((hist_target, hist_hours, future_target, future_hours))
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hist_target, hist_hours, future_target, future_hours = self._samples[idx]
        history_calendar = calendar_features(torch.from_numpy(hist_hours.copy()))
        decoder_calendar = calendar_features(torch.from_numpy(future_hours.copy()))
        
        return{
            "history_target": torch.from_numpy(hist_target.copy()).unsqueeze(-1),
            "history_calendar": history_calendar,
            "decoder_calendar": decoder_calendar,
            "future_target": torch.from_numpy(future_target.copy()).unsqueeze(-1)
        }
                