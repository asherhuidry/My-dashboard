"""Per-modality encoders — maps each data type to a fixed-dim embedding.

Each encoder takes raw domain data and outputs a single vector of size
`embed_dim` (default 128). These embeddings become the initial node features
fed into the graph attention network.

Modality → Encoder:
  Price/volume time series  → TemporalEncoder (LSTM + position-aware attention)
  Macro FRED series         → MacroEncoder    (transformer over normalized values)
  Fundamental ratios        → FundamentalEncoder (MLP with batch norm)
  Text / sentiment          → SentimentEncoder  (projection from keyword scores)

Design principle: every encoder output is L2-normalized so that distances in
embedding space are comparable across modalities. This is what lets the graph
attention layer meaningfully compare NVDA's price state to T10Y2Y's macro state.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Positional encoding ───────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017).

    Adds position information to a sequence so the model knows which
    timestep each feature vector corresponds to.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to x of shape (batch, seq_len, d_model)."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── 1. Temporal encoder (price + volume + technicals) ─────────────────────────

class TemporalEncoder(nn.Module):
    """Encodes a time series of features into a single embedding vector.

    Architecture: linear projection → LSTM → self-attention over hidden states
    → weighted pool → LayerNorm.

    The self-attention over LSTM hidden states lets the model focus on the
    most informative timesteps (e.g., the day of an earnings surprise) rather
    than naively taking only the last state.

    Args:
        input_size:  Number of features per timestep (e.g. 80 technical features).
        embed_dim:   Output embedding dimension.
        hidden_size: LSTM hidden state size.
        num_layers:  Number of LSTM layers.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        input_size:  int,
        embed_dim:   int  = 128,
        hidden_size: int  = 256,
        num_layers:  int  = 2,
        dropout:     float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
            bidirectional = True,
        )
        # Self-attention over LSTM outputs to pick important timesteps
        self.attn_query = nn.Linear(hidden_size * 2, 1)
        # Project bidirectional LSTM to embed_dim
        self.out_proj   = nn.Linear(hidden_size * 2, embed_dim)
        self.norm       = nn.LayerNorm(embed_dim)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode sequence.

        Args:
            x:    (batch, seq_len, input_size)
            mask: (batch, seq_len) bool mask — True = valid timestep.

        Returns:
            (batch, embed_dim) normalized embedding.
        """
        # Project input features to LSTM dim
        x = F.gelu(self.input_proj(x))      # (B, T, H)

        # LSTM encoding
        out, _ = self.lstm(x)               # (B, T, 2H) — bidirectional

        # Temporal attention: score each timestep's importance
        scores = self.attn_query(out).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)

        # Weighted sum over time
        context = (out * weights).sum(dim=1)   # (B, 2H)

        # Project to embedding space and normalize
        emb = self.out_proj(self.dropout(context))  # (B, embed_dim)
        return F.normalize(self.norm(emb), dim=-1)


# ── 2. Macro encoder (FRED series — single-value per timestamp) ───────────────

class MacroEncoder(nn.Module):
    """Encodes a macro time series (single float per day) to an embedding.

    For FRED series like T10Y2Y or VIX, each timestep is a single scalar.
    We augment each value with its Z-score, 21-day momentum, and regime flags,
    then encode the sequence with a lightweight Transformer.

    Args:
        seq_len:   Number of recent observations to use (e.g. 252 = 1Y daily).
        embed_dim: Output embedding dimension.
        d_model:   Transformer model dimension.
        n_heads:   Number of attention heads.
        n_layers:  Number of transformer layers.
    """

    def __init__(
        self,
        seq_len:   int   = 63,
        embed_dim: int   = 128,
        d_model:   int   = 64,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        # 4 augmented features: value, z_score, 21d_change, above_zero
        self.input_proj = nn.Linear(4, d_model)
        self.pe         = SinusoidalPE(d_model, max_len=seq_len + 1, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = n_heads,
            dim_feedforward = d_model * 4,
            dropout    = dropout,
            batch_first= True,
            norm_first = True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool        = nn.AdaptiveAvgPool1d(1)
        self.out_proj    = nn.Linear(d_model, embed_dim)
        self.norm        = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode macro series.

        Args:
            x: (batch, seq_len, 4) — [value, z_score, 21d_change, above_zero].

        Returns:
            (batch, embed_dim) normalized embedding.
        """
        h = F.gelu(self.input_proj(x))           # (B, T, d_model)
        h = self.pe(h)
        h = self.transformer(h)                   # (B, T, d_model)
        # Pool over time
        pooled = self.pool(h.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        emb    = self.out_proj(pooled)
        return F.normalize(self.norm(emb), dim=-1)


# ── 3. Fundamental encoder (financial ratios — tabular) ───────────────────────

class FundamentalEncoder(nn.Module):
    """Encodes a vector of fundamental ratios to an embedding.

    Uses a residual MLP with batch normalization. Handles NaN-filled inputs
    by masking them out (missing fundamentals = zero contribution).

    Args:
        input_size: Number of fundamental features (e.g. 20: P/E, P/S, margins, etc.).
        embed_dim:  Output embedding dimension.
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self,
        input_size: int   = 20,
        embed_dim:  int   = 128,
        hidden_dim: int   = 256,
        dropout:    float = 0.2,
    ) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Residual skip if dims match
        self.skip   = nn.Linear(input_size, hidden_dim)
        self.out    = nn.Linear(hidden_dim, embed_dim)
        self.norm   = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode fundamental ratios.

        Args:
            x: (batch, input_size) — NaN values replaced with 0 before passing.

        Returns:
            (batch, embed_dim) normalized embedding.
        """
        # Replace NaN with 0 (missing = unknown, not informative)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.block1(x) + self.skip(x)  # residual
        h = self.block2(h) + h             # residual
        emb = self.out(h)
        return F.normalize(self.norm(emb), dim=-1)


# ── 4. Sentiment encoder (social + options signals) ───────────────────────────

class SentimentEncoder(nn.Module):
    """Encodes social sentiment and options flow signals to an embedding.

    Input: a vector of sentiment signals (reddit_score, stocktwits_bull_pct,
    google_trends_zscore, put_call_ratio, fear_greed_index, ...).
    Output: embedding capturing the aggregate sentiment state.

    Args:
        input_size: Number of sentiment signals.
        embed_dim:  Output embedding dimension.
    """

    def __init__(
        self,
        input_size: int   = 8,
        embed_dim:  int   = 128,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sentiment signals.

        Args:
            x: (batch, input_size)

        Returns:
            (batch, embed_dim) normalized embedding.
        """
        x = torch.nan_to_num(x, nan=0.0)
        return F.normalize(self.net(x), dim=-1)


# ── 5. Multi-modal fusion ──────────────────────────────────────────────────────

class ModalFusion(nn.Module):
    """Fuse multiple modality embeddings into a single node embedding.

    Uses cross-modal attention: each modality attends to all others,
    learning which signals are most relevant for the current prediction.
    This is what allows "price momentum + credit stress + bearish Reddit"
    to combine more than naively concatenating them.

    Args:
        embed_dim:  Dimension of each modality embedding (must be equal).
        n_modalities: Number of modalities to fuse (2-4).
        n_heads:    Attention heads for cross-modal attention.
    """

    def __init__(
        self,
        embed_dim:    int   = 128,
        n_modalities: int   = 4,
        n_heads:      int   = 4,
        dropout:      float = 0.1,
    ) -> None:
        super().__init__()
        # Learnable modality type embeddings (so model knows WHICH modality it's reading)
        self.modal_tokens = nn.Parameter(torch.randn(n_modalities, embed_dim))

        # Transformer for cross-modal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = embed_dim,
            nhead      = n_heads,
            dim_feedforward = embed_dim * 4,
            dropout    = dropout,
            batch_first= True,
            norm_first = True,
        )
        self.cross_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out_proj   = nn.Linear(embed_dim * n_modalities, embed_dim)
        self.norm       = nn.LayerNorm(embed_dim)
        self.n_mod      = n_modalities

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Fuse modality embeddings.

        Args:
            embeddings: List of (batch, embed_dim) tensors, one per modality.
                        Missing modalities can be passed as zero tensors.

        Returns:
            (batch, embed_dim) fused embedding.
        """
        # Pad/truncate to expected number of modalities
        while len(embeddings) < self.n_mod:
            embeddings.append(torch.zeros_like(embeddings[0]))
        embeddings = embeddings[:self.n_mod]

        # Stack: (batch, n_mod, embed_dim)
        x = torch.stack(embeddings, dim=1)

        # Add modality-type tokens (tells the model which slot is which)
        x = x + self.modal_tokens.unsqueeze(0)

        # Cross-modal attention
        x = self.cross_attn(x)                        # (B, n_mod, embed_dim)

        # Flatten and project
        x = x.reshape(x.size(0), -1)                  # (B, n_mod * embed_dim)
        fused = self.out_proj(x)
        return F.normalize(self.norm(fused), dim=-1)   # (B, embed_dim)
