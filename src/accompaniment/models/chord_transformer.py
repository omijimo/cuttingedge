"""Compact encoder transformer: melody tokens  chord logits per timestep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .positional_encoding import SinusoidalPositionalEncoding


class ChordTransformer(nn.Module):
    def __init__(
        self,
        melody_vocab_size: int = 15,
        chord_vocab_size: int = 86,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        num_bar_positions: int = 16,
    ):
        super().__init__()
        self.d_model = d_model

        self.melody_embed = nn.Embedding(melody_vocab_size, d_model)
        self.beat_embed = nn.Embedding(num_bar_positions, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, chord_vocab_size)

    def forward(
        self,
        melody_tokens: torch.Tensor,
        beat_positions: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        melody_tokens : LongTensor [batch, seq_len]
        beat_positions : LongTensor [batch, seq_len]
        padding_mask : BoolTensor [batch, seq_len]  True = pad (ignored)

        Returns
        -------
        logits : FloatTensor [batch, seq_len, chord_vocab_size]
        """
        x = self.melody_embed(melody_tokens) + self.beat_embed(beat_positions)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return self.head(x)
