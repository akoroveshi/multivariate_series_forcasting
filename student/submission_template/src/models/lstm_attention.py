"""LSTM + Attention recurrent-attention hybrid forecaster.

Encoder-decoder architecture:
- Encoder -> multi-layer LSTM over the conditioning window
    (target history + known calendar covariates).
- Attention -> Bahdanau-style additive attention lets every decoder 
    step look back over all encoder hidden states,
    which helps mitigate vanishing gradients over long histories.
- Decoder -> LSTM that unrolls one rollout block (default 24 steps) at a
      time, autoregressively feeding back either the true previous value
      (teacher forcing) or its own last prediction (free running)

For horizons longer than one rollout block (e.g. the 336-step benchmark
horizon vs. a 24-step block), rollout re-invokes the decoder block by
block, feeding each block's predictions back in as additional history.
"""

from __future__ import annotations
import torch
from torch import nn

from .common import RevIN, SpatialDropout1d


class BahdanauAttention(nn.Module):
    """Additive attention over encoder hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.energy_proj = nn.Linear(hidden_size, 1)

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute an attention-weighted context vector """
        query_expanded = self.query_proj(query).unsqueeze(1)  # (B, 1, H)
        keys_proj = self.key_proj(keys)  # (B, S, H)
        energy = self.energy_proj(torch.tanh(query_expanded + keys_proj)).squeeze(-1)  # (B, S)
        weights = torch.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)  # (B, H)
        return context, weights


class LSTMAttentionForecaster(nn.Module):
    """Recurrent-attention hybrid forecaster """

    def __init__(
        self,
        num_calendar_features: int = 4,
        hidden_size: int = 128,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
        dropout: float = 0.1,
        block_len: int = 24,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_layers = decoder_layers
        self.block_len = block_len

        self.revin = RevIN(num_features=1, affine=True)
        self.input_dropout = SpatialDropout1d(dropout)

        encoder_input_size = 1 + num_calendar_features
        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=hidden_size,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0.0,
        )

        self.attention = BahdanauAttention(hidden_size)

        decoder_input_size = 1 + num_calendar_features + hidden_size
        self.decoder_cell = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_size,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout if decoder_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, 1)

        self.encoder_bridge = (
            nn.Linear(encoder_layers, decoder_layers)
            if encoder_layers != decoder_layers
            else None
        )

    def _bridge_state(self, state: torch.Tensor) -> torch.Tensor:
        """Reshape encoder final state (num_layers, B, H) to match the decoder's layer count."""
        if self.encoder_bridge is None:
            return state
        # (num_enc_layers, B, H) -> (B, H, num_enc_layers) -> (B, H, num_dec_layers) -> (num_dec_layers, B, H)
        reshaped = state.permute(1, 2, 0)
        reshaped = self.encoder_bridge(reshaped)
        return reshaped.permute(2, 0, 1).contiguous()

    def encode(
        self, history_target: torch.Tensor, history_calendar: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the encoder over the conditioning window """
        normalized_target = self.revin.normalize(history_target)
        
        encoder_in = torch.cat([normalized_target, history_calendar], dim=-1)
        encoder_in = self.input_dropout(encoder_in)
        encoder_outputs, (h_n, c_n) = self.encoder(encoder_in)
        return encoder_outputs, (self._bridge_state(h_n), self._bridge_state(c_n))

    def decode_block(
        self,
        last_target: torch.Tensor,
        decoder_calendar: torch.Tensor,
        encoder_outputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        future_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Autoregressively decode one rollout block (``block_len`` steps) """
        batch_size = last_target.size(0)
        h, c = state
        step_input_target = self.revin.normalize(last_target).squeeze(1)  
        predictions_normalized = []

        for step in range(decoder_calendar.size(1)):
            query = h[-1]  
            context, _ = self.attention(query, encoder_outputs, encoder_outputs)
            step_calendar = decoder_calendar[:, step, :]
            decoder_input = torch.cat([step_input_target, step_calendar, context], dim=-1).unsqueeze(1)
            
            _, (h, c) = self.decoder_cell(decoder_input, (h, c))
            step_pred_normalized = self.output_proj(h[-1])  
            predictions_normalized.append(step_pred_normalized)

            use_teacher_forcing = (
                self.training
                and future_target is not None
                and teacher_forcing_ratio > 0.0
                and torch.rand(batch_size, device=last_target.device).mean().item() < teacher_forcing_ratio
            )
            if use_teacher_forcing:
                step_input_target = self.revin.normalize(future_target[:, step : step + 1, :]).squeeze(1)
            else:
                step_input_target = step_pred_normalized

        predictions_normalized = torch.stack(predictions_normalized, dim=1)  
        predictions = self.revin.denormalize(predictions_normalized)
        return predictions, (h, c)

    def forward(
        self,
        history_target: torch.Tensor,
        history_calendar: torch.Tensor,
        decoder_calendar: torch.Tensor,
        future_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Predict one rollout block ahead of the conditioning window """
        encoder_outputs, state = self.encode(history_target, history_calendar)
        last_target = history_target[:, -1:, :]
        
        predictions, _ = self.decode_block(
            last_target,
            decoder_calendar,
            encoder_outputs,
            state,
            future_target=future_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return predictions

    @torch.no_grad()
    def rollout(
        self,
        history_target: torch.Tensor,
        history_calendar: torch.Tensor,
        decoder_calendar: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        """Autoregressively forecast horizon steps by chaining rollout blocks """
        self.eval()
        target = history_target
        calendar = history_calendar
        outputs = []
        remaining = horizon
        cursor = 0
        
        while remaining > 0:
            step = min(self.block_len, remaining)
            block_calendar = decoder_calendar[:, cursor : cursor + step, :]
            if step < self.block_len:
                pad = self.block_len - step
                block_calendar = torch.cat(
                    [block_calendar, block_calendar[:, -1:, :].expand(-1, pad, -1)], dim=1
                )
            block_pred = self.forward(target, calendar, block_calendar)
            block_pred = block_pred[:, :step, :]
            
            outputs.append(block_pred)
            target = torch.cat([target, block_pred], dim=1)
            calendar = torch.cat([calendar, decoder_calendar[:, cursor : cursor + step, :]], dim=1)
            cursor += step
            remaining -= step
            
        return torch.cat(outputs, dim=1)