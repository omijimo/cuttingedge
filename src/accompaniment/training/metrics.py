"""Evaluation metrics for chord prediction."""

from __future__ import annotations

import torch

from ..io.chord_vocab import NUM_QUALITIES, NO_CHORD_ID, PAD_CHORD_ID


def chord_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, pad_id: int = PAD_CHORD_ID
) -> float:
    mask = targets != pad_id
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()


def root_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, pad_id: int = PAD_CHORD_ID
) -> float:
    mask = (targets != pad_id) & (targets != NO_CHORD_ID)
    if mask.sum() == 0:
        return 0.0
    pred_roots = (preds[mask] // NUM_QUALITIES).clamp(0, 11)
    target_roots = (targets[mask] // NUM_QUALITIES).clamp(0, 11)
    return (pred_roots == target_roots).float().mean().item()


def quality_accuracy(
    preds: torch.Tensor, targets: torch.Tensor, pad_id: int = PAD_CHORD_ID
) -> float:
    mask = (targets != pad_id) & (targets != NO_CHORD_ID)
    if mask.sum() == 0:
        return 0.0
    pred_q = preds[mask] % NUM_QUALITIES
    target_q = targets[mask] % NUM_QUALITIES
    return (pred_q == target_q).float().mean().item()


def compute_metrics(
    preds: torch.Tensor, targets: torch.Tensor, pad_id: int = PAD_CHORD_ID
) -> dict[str, float]:
    return {
        "chord_accuracy": chord_accuracy(preds, targets, pad_id),
        "root_accuracy": root_accuracy(preds, targets, pad_id),
        "quality_accuracy": quality_accuracy(preds, targets, pad_id),
    }
