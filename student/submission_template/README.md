# Multivariate Time Series Forecasting

Implements the **LSTM + Attention** recurrent-attention hybrid and 
the **PatchTST** patch-based Transformer.

## Layout

```
predict.py              # inference entrypoint (required command)
requirements.txt
src/
  model.py               # ForecastModel:  wrapper + checkpoint (de)serialization
  dataset.py              # sliding-window dataset + calendar feature helpers
  train.py                # training loop (scheduled sampling + SGDR)
  models/
    common.py             # RevIN, SpatialDropout1d, calendar_features
    lstm_attention.py      # LSTMAttentionForecaster
    patchtst.py             # PatchTST
```

## Training

```bash
DATA_DIR=/path/to/downloaded/hf/dataset
python -m src.train \
  --train "$DATA_DIR/train.csv" \
  --forecast-index "$DATA_DIR/forecast_index_validation.csv" \
  --checkpoint-out checkpoint.pt \
  --history-len 168 --block-len 24 --epochs 30
```

`--forecast-index` is optional. When given each epoch also reports 
validation MAE by rolling the model
forward over the validation forecast index, and the checkpoint with the
best validation MAE is kept. 
`checkpoint.pt` stores both the weights and
the model config needed to rebuild it.

## Inference

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint /submission/checkpoint.pt
```

`input_dir` must contain `train.csv` (history) plus
`forecast_index_test.csv` or `forecast_index_validation.csv` (rows to
predict). Predictions are written with schema
`series_id,timestamp,prediction`, covering every requested row.

## Packaging

From inside this directory:

```bash
zip -r final_submission.zip predict.py requirements.txt checkpoint.pt src baselines.py
```