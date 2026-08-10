"""LSTM + Attention recurrent-attention hybrid forecaster.

Encoder-decoder architecture:

* **Encoder** -- multi-layer LSTM over the conditioning window, reading the
  RevIN-normalized target together with the covariates of those timesteps.
* **Attention** -- Bahdanau-style additive attention, so every decoder step can
  re-read all encoder states instead of relying on the single final state; this
  keeps gradients short over the 168-step conditioning window.
* **Decoder** -- LSTM that unrolls ``block_len`` steps, at each step consuming
  the *known* covariates of the step being predicted plus either the true
  previous target (teacher forcing) or its own previous prediction (free
  running). The mixture is annealed by scheduled sampling during training.

Horizons longer than one block (336 or 672 steps versus a 24-step block) are
covered by :func:`~src.models.common.chained_rollout`, which feeds each block's
predictions back in as additional history.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import RevIN, SeriesEmbedding, SpatialDropout1d, chained_rollout, inference_mode


class BahdanauAttention(nn.Module):
    """Additive attention over encoder hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.energy_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(context, weights)`` for a ``(B, H)`` query over ``(B, S, H)`` keys."""
        energy = self.energy_proj(
            torch.tanh(self.query_proj(query).unsqueeze(1) + self.key_proj(keys))
        ).squeeze(-1)
        weights = torch.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context, weights


class LSTMAttentionForecaster(nn.Module):
    """Recurrent-attention hybrid conditioned on known future covariates."""

    def __init__(
        self,
        num_history_features: int = 0,
        num_future_features: int | None = None,
        num_static_features: int = 0,
        hidden_size: int = 128,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        block_len: int = 24,
        history_len: int = 168,
        use_attention: bool = True,
        use_revin: bool = True,
        spatial_dropout: float = 0.1,
        num_series_slots: int = 1,
        series_embedding_dim: int = 16,
        **_: object,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.block_len = block_len
        self.history_len = history_len
        self.use_attention = use_attention
        self.use_revin = use_revin
        self.num_static_features = num_static_features

        self.revin = RevIN(num_features=1, affine=True) if use_revin else None
        self.input_dropout = SpatialDropout1d(spatial_dropout)

        num_future_features = (
            num_history_features if num_future_features is None else num_future_features
        )
        self.series_embedding = (
            SeriesEmbedding(num_series_slots, series_embedding_dim)
            if (num_series_slots > 1 and series_embedding_dim > 0)
            else None
        )
        static_size = num_static_features + (
            self.series_embedding.dim if self.series_embedding is not None else 0
        )
        encoder_input_size = 1 + num_history_features + static_size
        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
        )

        self.attention = BahdanauAttention(hidden_size) if use_attention else None
        context_size = hidden_size if use_attention else 0

        decoder_input_size = 1 + num_future_features + static_size + context_size
        self.decoder_cell = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout if decoder_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_size + context_size, 1)

        self.encoder_bridge = (
            nn.Linear(encoder_layers, decoder_layers) if encoder_layers != decoder_layers else None
        )

    # ------------------------------------------------------------------ utils
    def _bridge_state(self, state: torch.Tensor) -> torch.Tensor:
        """Map an encoder state ``(n_enc, B, H)`` onto the decoder's layer count."""
        if self.encoder_bridge is None:
            return state.contiguous()
        reshaped = self.encoder_bridge(state.permute(1, 2, 0))
        return reshaped.permute(2, 0, 1).contiguous()

    def _target_stats(self, history_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.revin is None:
            zeros = torch.zeros_like(history_target[:, :1, :])
            return zeros, torch.ones_like(zeros)
        return self.revin.stats(history_target)

    def _normalize(self, values: torch.Tensor, stats) -> torch.Tensor:
        return self.revin.normalize(values, stats) if self.revin is not None else values

    def _denormalize(self, values: torch.Tensor, stats) -> torch.Tensor:
        return self.revin.denormalize(values, stats) if self.revin is not None else values

    def _static_vector(
        self,
        static: torch.Tensor | None,
        series_index: torch.Tensor | None,
        batch_size: int,
        device,
    ) -> torch.Tensor | None:
        """Concatenate the static covariates with the learned series embedding."""
        parts = []
        if static is not None and static.numel():
            parts.append(static)
        if self.series_embedding is not None:
            parts.append(self.series_embedding(series_index, batch_size, device))
        return torch.cat(parts, dim=-1) if parts else None

    @staticmethod
    def _expand_static(static: torch.Tensor | None, length: int) -> torch.Tensor | None:
        if static is None or static.numel() == 0:
            return None
        return static.unsqueeze(1).expand(-1, length, -1)

    # ---------------------------------------------------------------- encoder
    def encode(
        self,
        history_target: torch.Tensor,
        history_features: torch.Tensor,
        static: torch.Tensor | None,
        stats,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        parts = [self._normalize(history_target, stats), history_features]
        static_expanded = self._expand_static(static, history_target.size(1))
        if static_expanded is not None:
            parts.append(static_expanded)
        encoder_in = self.input_dropout(torch.cat(parts, dim=-1))
        encoder_outputs, (h_n, c_n) = self.encoder(encoder_in)
        return encoder_outputs, (self._bridge_state(h_n), self._bridge_state(c_n))

    # ---------------------------------------------------------------- decoder
    def decode_block(
        self,
        last_target_normalized: torch.Tensor,
        future_features: torch.Tensor,
        static: torch.Tensor | None,
        encoder_outputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        stats,
        future_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Autoregressively decode one block and return raw-scale predictions."""
        batch_size, steps, _ = future_features.shape
        h, c = state
        step_target = last_target_normalized
        static_step = static if (static is not None and static.numel()) else None
        predictions: list[torch.Tensor] = []

        for step in range(steps):
            parts = [step_target, future_features[:, step, :]]
            if static_step is not None:
                parts.append(static_step)
            context = None
            if self.attention is not None:
                context, _ = self.attention(h[-1], encoder_outputs)
                parts.append(context)
            _, (h, c) = self.decoder_cell(torch.cat(parts, dim=-1).unsqueeze(1), (h, c))

            head_input = h[-1] if context is None else torch.cat([h[-1], context], dim=-1)
            step_prediction = self.output_proj(self.head_dropout(head_input))
            predictions.append(step_prediction)

            teacher = (
                self.training
                and future_target is not None
                and teacher_forcing_ratio > 0.0
                and step + 1 < steps
            )
            if teacher:
                truth = self._normalize(future_target[:, step : step + 1, :], stats).squeeze(1)
                keep = (
                    torch.rand(batch_size, 1, device=future_features.device) < teacher_forcing_ratio
                ).float()
                step_target = keep * truth + (1.0 - keep) * step_prediction
            else:
                step_target = step_prediction

        stacked = torch.stack(predictions, dim=1)
        return self._denormalize(stacked, stats)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        history_target: torch.Tensor,
        history_features: torch.Tensor,
        future_features: torch.Tensor,
        static: torch.Tensor | None = None,
        future_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        series_index: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        """Predict ``future_features.size(1)`` steps ahead of the history window."""
        stats = self._target_stats(history_target)
        static_vector = self._static_vector(
            static, series_index, history_target.size(0), history_target.device
        )
        encoder_outputs, state = self.encode(
            history_target, history_features, static_vector, stats
        )
        last_target = self._normalize(history_target[:, -1:, :], stats).squeeze(1)
        return self.decode_block(
            last_target,
            future_features,
            static_vector,
            encoder_outputs,
            state,
            stats,
            future_target=future_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

    @torch.no_grad()
    def rollout(
        self,
        history_target: torch.Tensor,
        history_features: torch.Tensor,
        future_features: torch.Tensor,
        static: torch.Tensor | None = None,
        horizon: int | None = None,
        future_history_features: torch.Tensor | None = None,
        series_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast ``horizon`` steps by chaining ``block_len``-step blocks."""
        horizon = future_features.size(1) if horizon is None else horizon
        with inference_mode(self):
            return chained_rollout(
                lambda y, x, f: self.forward(y, x, f, static=static, series_index=series_index),
                history_target,
                history_features,
                future_features,
                horizon=horizon,
                block_len=self.block_len,
                history_len=history_target.size(1),
                future_history_features=future_history_features,
            )
