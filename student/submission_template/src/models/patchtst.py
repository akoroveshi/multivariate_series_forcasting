"""PatchTST-style patch Transformer with known-future covariate tokens.

The backbone follows PatchTST (Nie et al., 2023): the target history is cut into
overlapping sub-series *patches*, each patch is linearly embedded into one token,
and a Transformer encoder attends over the token sequence. Patching keeps local
semantics inside a token and shrinks attention from ``O(L^2)`` to
``O((L/S)^2)``. Channel independence is preserved in the sense of the paper --
one shared backbone processes a single univariate target channel, and every
series is an independent sample.

Two extensions adapt the backbone to this benchmark, where all 22 covariates are
known for the forecast window:

``covariate_mode="tokens"`` (default)
    Future covariates are patched as well and appended to the token sequence as
    *query tokens*. Self-attention therefore relates each forecast segment both
    to the observed history and to the other forecast segments, and the head
    reads the horizon straight off the query tokens. This yields a **direct**
    multi-step forecast, i.e. no autoregressive error accumulation inside a
    block.

``covariate_mode="none"``
    Vanilla PatchTST: target-only tokens plus a flatten-linear head. Used as an
    ablation to isolate the contribution of the covariates.

History covariates are always fused into the history tokens (a patch of the
covariate matrix is flattened and projected), and static per-series features are
added to every token.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import RevIN, SeriesEmbedding, chained_rollout, inference_mode


class PatchTST(nn.Module):
    """Patch Transformer forecaster with optional future-covariate tokens."""

    def __init__(
        self,
        num_history_features: int = 0,
        num_future_features: int | None = None,
        num_static_features: int = 0,
        history_len: int = 168,
        block_len: int = 336,
        patch_len: int = 24,
        stride: int = 12,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        head_dropout: float = 0.0,
        covariate_mode: str = "tokens",
        use_revin: bool = True,
        num_series_slots: int = 1,
        series_embedding_dim: int = 16,
        pointwise_head: bool = True,
        pointwise_hidden: int = 64,
        **_: object,
    ) -> None:
        super().__init__()
        if covariate_mode not in {"tokens", "none"}:
            raise ValueError("covariate_mode must be 'tokens' or 'none'.")
        if patch_len > history_len:
            raise ValueError("patch_len cannot exceed history_len.")
        if covariate_mode == "tokens" and block_len % patch_len != 0:
            raise ValueError("block_len must be a multiple of patch_len when using covariate tokens.")

        self.history_len = history_len
        self.block_len = block_len
        self.patch_len = patch_len
        self.stride = stride
        self.covariate_mode = covariate_mode
        num_future_features = (
            num_history_features if num_future_features is None else num_future_features
        )
        self.num_history_features = num_history_features
        self.num_future_features = num_future_features
        self.num_static_features = num_static_features

        self.num_history_patches = (history_len - patch_len) // stride + 1
        self.num_future_patches = block_len // patch_len if covariate_mode == "tokens" else 0

        self.revin = RevIN(num_features=1, affine=True) if use_revin else None

        self.target_embed = nn.Linear(patch_len, d_model)
        self.history_covariate_embed = (
            nn.Linear(patch_len * num_history_features, d_model) if num_history_features else None
        )
        self.future_covariate_embed = (
            nn.Linear(patch_len * num_future_features, d_model)
            if (covariate_mode == "tokens" and num_future_features)
            else None
        )
        self.future_token = (
            nn.Parameter(torch.zeros(1, 1, d_model)) if covariate_mode == "tokens" else None
        )
        self.series_embedding = (
            SeriesEmbedding(num_series_slots, series_embedding_dim)
            if (num_series_slots > 1 and series_embedding_dim > 0)
            else None
        )
        static_size = num_static_features + (
            self.series_embedding.dim if self.series_embedding is not None else 0
        )
        self.static_size = static_size
        self.static_embed = nn.Linear(static_size, d_model) if static_size else None

        # Pointwise residual path. A patch token compresses ``patch_len`` hours of
        # covariates into one d_model vector before the head expands it back into
        # ``patch_len`` predictions, which throws away within-patch resolution. The
        # benchmark's drivers act largely instantaneously (the strongest covariate
        # correlates 0.68 with the target at the *same* hour), so a small MLP applied
        # per forecast step restores that resolution and the Transformer only has to
        # model the residual structure.
        self.pointwise = (
            nn.Sequential(
                nn.Linear(num_future_features + static_size, pointwise_hidden),
                nn.GELU(),
                nn.Linear(pointwise_hidden, 1),
            )
            if (pointwise_head and covariate_mode == "tokens" and num_future_features)
            else None
        )

        total_tokens = self.num_history_patches + self.num_future_patches
        self.position = nn.Parameter(torch.zeros(1, total_tokens, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.embed_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(head_dropout)
        if covariate_mode == "tokens":
            self.head = nn.Linear(d_model, patch_len)
        else:
            self.head = nn.Linear(self.num_history_patches * d_model, block_len)

    # ------------------------------------------------------------------ utils
    def _patchify(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """``(B, T, C)`` -> ``(B, N, patch_len * C)`` sliding patches."""
        patches = x.unfold(dimension=1, size=self.patch_len, step=stride)  # (B, N, C, P)
        return patches.permute(0, 1, 3, 2).flatten(start_dim=2)

    def _target_stats(self, history_target: torch.Tensor):
        if self.revin is None:
            zeros = torch.zeros_like(history_target[:, :1, :])
            return zeros, torch.ones_like(zeros)
        return self.revin.stats(history_target)

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
        """Direct multi-step forecast for ``block_len`` steps (raw target scale)."""
        del future_target, teacher_forcing_ratio  # direct model: no rollout inside a block
        stats = self._target_stats(history_target)
        normalized = (
            self.revin.normalize(history_target, stats) if self.revin is not None else history_target
        )
        static_vector = self._static_vector(
            static, series_index, history_target.size(0), history_target.device
        )

        tokens = self.target_embed(self._patchify(normalized, self.stride))
        if self.history_covariate_embed is not None:
            tokens = tokens + self.history_covariate_embed(
                self._patchify(history_features, self.stride)
            )

        future = None
        if self.covariate_mode == "tokens":
            future = future_features[:, : self.block_len, :]
            if future.size(1) < self.block_len:
                pad = self.block_len - future.size(1)
                future = torch.cat([future, future[:, -1:, :].expand(-1, pad, -1)], dim=1)
            query = self.future_token.expand(future.size(0), self.num_future_patches, -1)
            if self.future_covariate_embed is not None:
                query = query + self.future_covariate_embed(self._patchify(future, self.patch_len))
            tokens = torch.cat([tokens, query], dim=1)

        if self.static_embed is not None and static_vector is not None:
            tokens = tokens + self.static_embed(static_vector).unsqueeze(1)

        encoded = self.norm(self.encoder(self.embed_dropout(tokens + self.position)))

        if self.covariate_mode == "tokens":
            future_states = encoded[:, self.num_history_patches :, :]
            prediction = self.head(self.head_dropout(future_states))
            prediction = prediction.reshape(prediction.size(0), -1, 1)
        else:
            flat = encoded[:, : self.num_history_patches, :].flatten(start_dim=1)
            prediction = self.head(self.head_dropout(flat)).unsqueeze(-1)

        if self.pointwise is not None and future is not None:
            parts = [future]
            if static_vector is not None:
                parts.append(static_vector.unsqueeze(1).expand(-1, future.size(1), -1))
            prediction = prediction + self.pointwise(torch.cat(parts, dim=-1))

        if self.revin is not None:
            prediction = self.revin.denormalize(prediction, stats)
        return prediction

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
        """Cover ``horizon`` steps; longer horizons chain ``block_len``-step blocks."""
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
