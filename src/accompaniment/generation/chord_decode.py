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

def sample_decode(
    logits: torch.Tensor,
    grid_resolution: int = 16,
    steps_per_beat: int = 4,
    temperature: float = 1.2,
    top_k: int = 5,
    avoid_repeat: bool = True,
) -> list[int]:
    """Sample chord ids per beat instead of always taking argmax."""
    if logits.dim() == 3:
        logits = logits[0]

    seq_len = logits.size(0)
    chords: list[int] = []
    prev_chord = None

    for start in range(0, seq_len, steps_per_beat):
        end = min(start + steps_per_beat, seq_len)

        # average logits within this beat
        beat_logits = logits[start:end].mean(dim=0)

        # do not sample PAD
        beat_logits[PAD_CHORD_ID] = -1e9

        # temperature
        beat_logits = beat_logits / temperature

        # top-k sampling
        values, indices = torch.topk(beat_logits, k=top_k)
        probs = torch.softmax(values, dim=-1)

        # optional: reduce immediate repetition
        if avoid_repeat and prev_chord is not None:
            for i, cid in enumerate(indices):
                if int(cid) == int(prev_chord):
                    probs[i] *= 0.4
            probs = probs / probs.sum()

        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        chord_id = int(indices[sampled_idx])

        chords.append(chord_id)
        prev_chord = chord_id

    return chords

def decode_to_labels(chord_ids: list[int]) -> list[str]:
    return [chord_to_string(cid) for cid in chord_ids]
