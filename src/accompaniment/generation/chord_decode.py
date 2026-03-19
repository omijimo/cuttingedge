"""Decode chord model output to per-beat chord labels."""

from __future__ import annotations

import numpy as np
import torch

from ..io.chord_vocab import PAD_CHORD_ID, chord_to_string


def greedy_decode(
    logits: torch.Tensor,
    grid_resolution: int = 16,
    steps_per_beat: int = 4,
) -> list[int]:
    """Take argmax per timestep, then majority-vote within each beat.

    Parameters
    ----------
    logits : [1, seq_len, V] or [seq_len, V]

    Returns
    -------
    chord_ids : one int per beat
    """
    if logits.dim() == 3:
        logits = logits[0]

    preds = logits.argmax(dim=-1).cpu().numpy()
    seq_len = len(preds)

    chords: list[int] = []
    for start in range(0, seq_len, steps_per_beat):
        end = min(start + steps_per_beat, seq_len)
        beat_preds = preds[start:end]
        valid = beat_preds[beat_preds != PAD_CHORD_ID]
        if len(valid) == 0:
            chords.append(PAD_CHORD_ID)
        else:
            vals, counts = np.unique(valid, return_counts=True)
            chords.append(int(vals[counts.argmax()]))
    return chords


def decode_to_labels(chord_ids: list[int]) -> list[str]:
    return [chord_to_string(cid) for cid in chord_ids]
