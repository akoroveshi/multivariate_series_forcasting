# Model package — Group 140

Multivariate time series forecasting for the Deep Learning (SS26) bonus project.
This directory *is* the submission archive: zipping `predict.py`,
`requirements.txt`, `checkpoint.pt` and `src/` produces `final_submission.zip`.

See the [repository README](../../README.md) for the full reproduction pipeline,
the evaluation protocol and the report.

## Layout

```
predict.py                 # required inference entrypoint
requirements.txt
src/
  features.py              # covariate groups, missing-value handling, standardisation, panels
  dataset.py               # sliding-window dataset + inference batching
  models/
    common.py              # RevIN (stateless), SpatialDropout1d, SeriesEmbedding, chained_rollout
    patchtst.py            # PatchTST-cov: patch Transformer + future covariate query tokens
    lstm_attention.py      # LSTM encoder-decoder with Bahdanau attention
    ensemble.py            # weighted blend of trained members
  model.py                 # ForecastModel: checkpoint (de)serialisation + bundled context
  metrics.py               # leaderboard metrics (WAPE primary) and training losses
  train.py                 # training / model-selection entrypoint
  evaluate.py              # rolling forecast generation and scoring
  baselines.py             # naive, lag and seasonal reference forecasts
```

Every model exposes the same two entry points:

```python
forward(history_target, history_features, future_features, static=None,
        series_index=None, future_target=None, teacher_forcing_ratio=0.0)  # (B, block_len, 1)
rollout(history_target, history_features, future_features, static=None,
        series_index=None, horizon=None, future_history_features=None)     # (B, horizon, 1)
```

## Training

```bash
DATA_DIR=../../data

# PatchTST-cov: direct 336-step decoding
python -m src.train --train "$DATA_DIR/train.csv" --model patchtst \
  --history-len 168 --block-len 336 --stride 12 \
  --epochs 24 --sgdr-t0 8 --sgdr-tmult 1 --patience 8 \
  --checkpoint-out ../../runs/patchtst_main/checkpoint.pt \
  --metrics-out ../../results/patchtst_main.json

# LSTM+Attention: iterative 24-step blocks with scheduled sampling
python -m src.train --train "$DATA_DIR/train.csv" --model lstm_attention \
  --history-len 168 --block-len 24 --stride 24 \
  --epochs 15 --sgdr-t0 5 --sgdr-tmult 1 --patience 6 \
  --teacher-forcing-start 0.9 --teacher-forcing-end 0.1 \
  --checkpoint-out ../../runs/lstm_main/checkpoint.pt \
  --metrics-out ../../results/lstm_main.json
```

`--val-len` / `--test-len` (default `336` each) carve two held-out windows off the
end of `train.csv`. The first is used for early stopping and every hyperparameter
choice; the second starts 336 steps after the conditioning window and therefore
reproduces the private test split's unobserved gap. Set both to `0` to train the
final model on the entire public history.

Useful flags: `--no-revin`, `--no-attention`, `--no-covariates`,
`--no-series-embedding`, `--no-pointwise-head`, `--covariate-mode none`,
`--loss {l1,mse,huber,wape}`, `--eval-protocol {rollout,windows}`. Run
`python -m src.train --help` for the full list.

**Both shipped models pass `--no-revin`.** RevIN is implemented and is the default,
but the ablation study found it harmful at this horizon — and severely so for the
iterative decoder, whose chained blocks re-estimate the instance statistics from
their own predictions. See the results section of the [repository
README](../../README.md).

The checkpoint stores the weights, the architecture config, the feature
specification, the covariate standardisation statistics and the
`series_id -> embedding row` mapping, so inference needs no training CSV.

## Inference

```bash
python predict.py --input_dir /data/input --output_file /output/predictions.csv \
  --checkpoint /submission/checkpoint.pt
```

`input_dir` must contain a forecast index (`forecast_index_test.csv` or
`forecast_index_validation.csv`) and at least one covariate table
(`test_input.csv`, `validation_input.csv`, `train.csv` or `history.csv`).
Output schema is `series_id,timestamp,prediction`, one row per forecast-index row,
in the input order.

Because the private input directory carries no observed target and its window
starts 336 steps after the last released one, the checkpoint bundles a *context
table*: a 504-step tail of the public history (with targets) plus the public
validation covariates. `predict.py` merges bundled context with whatever the input
directory provides (the input directory wins on duplicate keys), reconstructs one
continuous per-series timeline, and rolls the model forward from the last observed
target to the final requested timestamp. No network access is required.

## Packaging

From the repository root:

```bash
python ../../scripts/make_submission.py --retrain
python ../../scripts/verify_submission.py
```
