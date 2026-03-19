"""Collation for DataLoader."""

import torch


def chord_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}
