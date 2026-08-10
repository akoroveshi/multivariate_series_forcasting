"""Self-contained tests for the forecasting package.

Runs with plain Python (``python tests/test_pipeline.py``) or under pytest.
Every test builds its own synthetic panel, so no dataset download is needed.

Several of these are regression tests for defects that were present in the code
we started from: the cached RevIN statistics, the missing PatchTST module, and the
inverted history-length guard in the evaluation loop.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "student" / "submission_template"
sys.path.insert(0, str(PKG))

from src.dataset import WindowDataset, build_inference_batch  # noqa: E402
from src.evaluate import evaluate_split, evaluate_windows, forecast_frame  # noqa: E402
from src.features import FeatureSpec, SeriesPanel, split_positions  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.model import ForecastModel  # noqa: E402
from src.models.common import RevIN, SpatialDropout1d, chained_rollout  # noqa: E402

HISTORY, BLOCK, LENGTH, SERIES = 48, 12, 400, 4
DYNAMIC = ("hour_sin", "hour_cos", "driver_a", "driver_b")
STATIC = ("capacity",)


def synthetic_frame(seed: int = 0, with_missing: bool = True) -> pd.DataFrame:
    """A small multi-series panel with known-future drivers and a few gaps."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range("2024-01-01", periods=LENGTH, freq="h")
    rows = []
    for index in range(SERIES):
        hour = stamps.hour.to_numpy()
        driver_a = rng.normal(size=LENGTH).cumsum() / 20.0
        driver_b = rng.normal(size=LENGTH)
        target = (
            5.0
            + index
            + 2.0 * np.sin(2 * np.pi * hour / 24)
            + 1.5 * driver_a
            + 0.4 * driver_b
            + rng.normal(scale=0.2, size=LENGTH)
        )
        frame = pd.DataFrame(
            {
                "series_id": f"unit_{index:02d}",
                "timestamp": stamps,
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "driver_a": driver_a,
                "driver_b": driver_b,
                "capacity": 10.0 + index,
                "target": target,
            }
        )
        if with_missing:
            holes = rng.choice(LENGTH, size=LENGTH // 20, replace=False)
            frame.loc[holes, "driver_a"] = np.nan
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def make_panel(**kwargs) -> SeriesPanel:
    spec = FeatureSpec(dynamic=DYNAMIC, static=STATIC, mask_features=("driver_a",))
    return SeriesPanel.from_frame(synthetic_frame(**kwargs), spec)


def make_model(panel: SeriesPanel, model_type: str, block_len: int = BLOCK, **extra) -> ForecastModel:
    config = dict(
        model_type=model_type,
        num_history_features=panel.num_history_features,
        num_future_features=panel.num_future_features,
        num_static_features=panel.num_static_features,
        history_len=HISTORY,
        block_len=block_len,
        num_series_slots=panel.num_series_slots,
        series_embedding_dim=8,
    )
    if model_type == "patchtst":
        config.update(patch_len=6, stride=6, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    else:
        config.update(hidden_size=32, encoder_layers=2, decoder_layers=1)
    config.update(extra)
    model = ForecastModel(**config)
    model.attach_preprocessing(panel.spec, panel.scaler)
    model.attach_series_index(panel.series_index)
    model.eval()
    return model


# --------------------------------------------------------------------- features


def test_missing_values_become_indicators() -> None:
    """Gaps are interpolated and their positions preserved as extra features."""
    panel = make_panel()
    assert panel.num_history_features == len(DYNAMIC) + 1  # + driver_a mask
    assert panel.num_future_features == panel.num_history_features
    arrays = panel["unit_00"]
    assert np.isfinite(arrays.x).all(), "imputation must leave no NaNs"
    mask_column = panel.spec.history_names.index("driver_a__isna")
    assert arrays.x[:, mask_column].sum() > 0, "the missingness indicator must fire"


def test_past_only_features_narrow_the_decoder_input() -> None:
    spec = FeatureSpec(
        dynamic=("hour_sin", "hour_cos"), past_only=("driver_a", "driver_b"), static=STATIC
    )
    panel = SeriesPanel.from_frame(synthetic_frame(with_missing=False), spec)
    assert panel.num_history_features == 4
    assert panel.num_future_features == 2
    assert tuple(panel.past_only_index) == (2, 3)


def test_past_only_columns_are_not_leaked_into_the_rollout() -> None:
    """Encoder rows for future steps must repeat the last observation, not the truth."""
    spec = FeatureSpec(dynamic=("hour_sin",), past_only=("driver_a",), static=STATIC)
    panel = SeriesPanel.from_frame(synthetic_frame(with_missing=False), spec)
    batch = build_inference_batch(panel, ["unit_00"], history_end=200, history_len=HISTORY, horizon=24)
    column = panel.past_only_index[0]
    filled = batch["future_history_features"][0, :, column]
    assert torch.allclose(filled, filled[0].expand_as(filled)), "past-only column must be constant"
    assert torch.isclose(filled[0], batch["history_features"][0, -1, column])
    truth = torch.tensor(panel["unit_00"].x[200:224, column])
    assert not torch.allclose(filled, truth), "true future values must not appear"


def test_window_dataset_skips_windows_without_targets() -> None:
    panel = make_panel()
    arrays = panel["unit_01"]
    arrays.y = arrays.y.copy()
    arrays.y[100:110] = np.nan
    dataset = WindowDataset(panel, history_len=HISTORY, horizon=BLOCK, stride=1)
    for index in range(len(dataset)):
        item = dataset[index]
        assert torch.isfinite(item["history_target"]).all()
        assert torch.isfinite(item["future_target"]).all()
    assert len(dataset) > 0


# ----------------------------------------------------------------------- RevIN


def test_revin_round_trip_is_identity() -> None:
    revin = RevIN(num_features=1, affine=True)
    x = torch.randn(3, 40, 1) * 4 + 7
    stats = revin.stats(x)
    restored = revin.denormalize(revin.normalize(x, stats), stats)
    assert torch.allclose(x, restored, atol=1e-4)


def test_revin_is_stateless() -> None:
    """Regression test: normalising one step must not corrupt the window statistics.

    The template cached ``mean``/``std`` on the module, so calling ``normalize`` on
    a single timestep inside the decoder replaced them with ``(y_t, 0)`` and the
    subsequent de-normalisation undid the wrong transform.
    """
    revin = RevIN(num_features=1, affine=False)
    window = torch.randn(2, 50, 1) * 3 + 10
    stats = revin.stats(window)
    revin.normalize(window[:, -1:, :], revin.stats(window[:, -1:, :]))  # the poisoning call
    restored = revin.denormalize(revin.normalize(window, stats), stats)
    assert torch.allclose(window, restored, atol=1e-4)


def test_spatial_dropout_is_identity_in_eval() -> None:
    dropout = SpatialDropout1d(0.5).eval()
    x = torch.randn(2, 10, 4)
    assert torch.allclose(dropout(x), x)


# ---------------------------------------------------------------------- models


def test_model_shapes_and_rollout() -> None:
    panel = make_panel()
    for model_type in ("patchtst", "lstm_attention"):
        model = make_model(panel, model_type)
        dataset = WindowDataset(panel, history_len=HISTORY, horizon=BLOCK, stride=17)
        batch = torch.utils.data.default_collate([dataset[i] for i in range(4)])
        out = model(
            batch["history_target"],
            batch["history_features"],
            batch["future_features"],
            static=batch["static"],
            series_index=batch["series_index"],
        )
        assert out.shape == (4, BLOCK, 1), (model_type, out.shape)
        assert torch.isfinite(out).all()

        horizon = 5 * BLOCK
        inference = build_inference_batch(panel, list(panel.keys()), 200, HISTORY, horizon)
        rolled = model.rollout(
            inference["history_target"],
            inference["history_features"],
            inference["future_features"],
            static=inference["static"],
            series_index=inference["series_index"],
            horizon=horizon,
            future_history_features=inference["future_history_features"],
        )
        assert rolled.shape == (SERIES, horizon, 1), (model_type, rolled.shape)
        assert torch.isfinite(rolled).all()


def test_single_block_rollout_matches_forward() -> None:
    """With ``horizon == block_len`` the rollout must not change the prediction."""
    panel = make_panel()
    for model_type in ("patchtst", "lstm_attention"):
        model = make_model(panel, model_type)
        batch = build_inference_batch(panel, list(panel.keys()), 200, HISTORY, BLOCK)
        with torch.no_grad():
            direct = model(
                batch["history_target"],
                batch["history_features"],
                batch["future_features"],
                static=batch["static"],
                series_index=batch["series_index"],
            )
            rolled = model.rollout(
                batch["history_target"],
                batch["history_features"],
                batch["future_features"],
                static=batch["static"],
                series_index=batch["series_index"],
                horizon=BLOCK,
                future_history_features=batch["future_history_features"],
            )
        assert torch.allclose(direct, rolled, atol=1e-5), model_type


def test_chained_rollout_respects_the_history_window() -> None:
    """The conditioning window must stay at ``history_len`` while chaining."""
    seen: list[int] = []

    def predict_block(y, x, f):
        seen.append(y.size(1))
        return torch.zeros(y.size(0), f.size(1), 1)

    chained_rollout(
        predict_block,
        torch.zeros(2, HISTORY, 1),
        torch.zeros(2, HISTORY, 3),
        torch.zeros(2, 5 * BLOCK, 3),
        horizon=5 * BLOCK,
        block_len=BLOCK,
        history_len=HISTORY,
    )
    assert seen == [HISTORY] * 5, seen


def test_series_embedding_falls_back_for_unknown_series() -> None:
    panel = make_panel()
    model = make_model(panel, "patchtst")
    batch = build_inference_batch(panel, list(panel.keys()), 200, HISTORY, BLOCK)
    with torch.no_grad():
        known = model(
            batch["history_target"], batch["history_features"], batch["future_features"],
            static=batch["static"], series_index=batch["series_index"],
        )
        unknown = model(
            batch["history_target"], batch["history_features"], batch["future_features"],
            static=batch["static"], series_index=torch.zeros_like(batch["series_index"]),
        )
    assert torch.isfinite(unknown).all(), "row 0 must be a usable fallback"
    assert not torch.allclose(known, unknown), "the embedding must actually be used"


# --------------------------------------------------------------------- metrics


def test_metrics_match_the_leaderboard_formulas() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.5, 2.0, 2.0, 5.0])
    metrics = compute_metrics(y, p)
    assert np.isclose(metrics["mae"], 0.625)
    assert np.isclose(metrics["mse"], 0.5625)
    assert np.isclose(metrics["rmse"], 0.75)
    assert np.isclose(metrics["wape"], 2.5 / 10.0 * 100.0)


# ----------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_reproduces_predictions() -> None:
    panel = make_panel()
    fit, val, _ = split_positions(LENGTH, 60, 60)
    for model_type in ("patchtst", "lstm_attention"):
        model = make_model(panel, model_type, clip_min=0.0)
        before, _ = evaluate_split(model, panel, val, fit.stop, "cpu", HISTORY, clip_min=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "checkpoint.pt")
            model.save_checkpoint(path)
            restored = ForecastModel.load_checkpoint(path)
        assert restored.series_index == model.series_index
        assert restored.feature_spec.history_names == panel.spec.history_names
        after, _ = evaluate_split(restored, panel, val, fit.stop, "cpu", HISTORY, clip_min=0.0)
        assert np.isclose(before["wape"], after["wape"], atol=1e-6), model_type


def test_ensemble_blends_its_members() -> None:
    panel = make_panel()
    members = [
        make_model(panel, "patchtst", clip_min=0.0).config,
        make_model(panel, "lstm_attention", clip_min=0.0).config,
    ]
    ensemble = ForecastModel(
        model_type="ensemble", members=members, member_weights=[0.7, 0.3],
        history_len=HISTORY, clip_min=0.0,
    )
    ensemble.attach_preprocessing(panel.spec, panel.scaler)
    ensemble.attach_series_index(panel.series_index)
    assert np.isclose(ensemble.net.weights.tolist(), [0.7, 0.3]).all()

    batch = build_inference_batch(panel, list(panel.keys()), 200, HISTORY, 2 * BLOCK)
    args = (batch["history_target"], batch["history_features"], batch["future_features"])
    kwargs = dict(
        static=batch["static"], series_index=batch["series_index"], horizon=2 * BLOCK,
        future_history_features=batch["future_history_features"],
    )
    with torch.no_grad():
        blended = ensemble.rollout(*args, **kwargs)
        parts = [member.rollout(*args, **kwargs) for member in ensemble.net.members]
    assert torch.allclose(blended, 0.7 * parts[0] + 0.3 * parts[1], atol=1e-5)


# ------------------------------------------------------------------ evaluation


def test_forecast_frame_covers_every_requested_row() -> None:
    panel = make_panel()
    model = make_model(panel, "patchtst", clip_min=0.0)
    spec = panel.spec
    index = pd.concat(
        [
            pd.DataFrame({spec.series_col: sid, spec.time_col: arrays.timestamps[300:340]})
            for sid, arrays in panel.items()
        ],
        ignore_index=True,
    )
    frame = forecast_frame(
        model, panel, index, "cpu", HISTORY, clip_min=0.0,
        history_end={sid: 250 for sid in panel.keys()},  # forces a 50-step gap
    )
    assert len(frame) == len(index)
    assert frame["prediction"].notna().all()
    assert np.isfinite(frame["prediction"]).all()


def test_windowed_evaluation_runs_over_several_origins() -> None:
    panel = make_panel()
    model = make_model(panel, "patchtst", clip_min=0.0)
    metrics = evaluate_windows(
        model, panel, slice(200, 380), history_len=HISTORY, horizon=BLOCK,
        stride=BLOCK, device="cpu", clip_min=0.0,
    )
    assert set(metrics) >= {"mae", "wape"}
    assert np.isfinite(metrics["mae"])


def test_predict_entrypoint_end_to_end() -> None:
    """The required inference command must run and cover the whole forecast index."""
    import subprocess

    panel = make_panel(with_missing=False)
    model = make_model(panel, "patchtst", clip_min=0.0)
    frame = synthetic_frame(with_missing=False)
    covariates = ["series_id", "timestamp", *panel.spec.covariates, *panel.spec.static]
    history = frame.groupby("series_id", sort=False).head(LENGTH - 40)
    future = frame.groupby("series_id", sort=False).tail(40)[covariates]
    model.attach_context(history)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "input"
        input_dir.mkdir()
        future.to_csv(input_dir / "test_input.csv", index=False)
        future[["series_id", "timestamp"]].to_csv(input_dir / "forecast_index_test.csv", index=False)
        checkpoint = tmp_dir / "checkpoint.pt"
        model.save_checkpoint(str(checkpoint))
        output = tmp_dir / "predictions.csv"
        subprocess.run(
            [
                sys.executable, "predict.py",
                "--input_dir", str(input_dir),
                "--output_file", str(output),
                "--checkpoint", str(checkpoint),
            ],
            cwd=PKG, check=True, capture_output=True,
        )
        predictions = pd.read_csv(output)
    assert list(predictions.columns) == ["series_id", "timestamp", "prediction"]
    assert len(predictions) == len(future)
    assert not predictions.duplicated(["series_id", "timestamp"]).any()
    assert np.isfinite(predictions["prediction"]).all()


def main() -> int:
    torch.manual_seed(0)
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as error:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"FAIL  {test.__name__}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
