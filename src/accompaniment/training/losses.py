"""Loss functions for chord prediction."""

import torch
import torch.nn as nn


class ChordPredictionLoss(nn.Module):
    def __init__(self, pad_id: int = 85, label_smoothing: float = 0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=label_smoothing,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: [B, T, V]  targets: [B, T]"""
        B, T, V = logits.shape
        return self.ce(logits.reshape(B * T, V), targets.reshape(B * T))
