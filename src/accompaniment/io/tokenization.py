"""Melody tokenization for the chord-prediction model.

Token vocabulary (MELODY_VOCAB_SIZE = 15):
    0-11 : pitch class  (C=0 … B=11) — used for note-on events
    12   : HOLD  — sustain of previous note
    13   : REST  — silence
    14   : PAD   — padding for batching
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

# ---- vocabulary constants ------------------------------------------------

PITCH_CLASS_OFFSET = 0
HOLD_TOKEN = 12
REST_TOKEN = 13
PAD_TOKEN = 14
MELODY_VOCAB_SIZE = 15

MAX_BAR_POSITIONS = 16  # 16th-note grid in 4/4


# ---- data carrier --------------------------------------------------------

class MelodyEvent(NamedTuple):
    pitch: int       # MIDI pitch 0-127, or -1 for rest/hold
    state: str       # "on" | "hold" | "rest"
    bar: int         # bar index (0-based)
    position: int    # position within bar (0 .. grid_resolution-1)


# ---- tokenization --------------------------------------------------------

def melody_event_to_token(event: MelodyEvent) -> int:
    if event.state == "rest":
        return REST_TOKEN
    if event.state == "hold":
        return HOLD_TOKEN
    return event.pitch % 12


def tokenize_melody(
    events: list[MelodyEvent],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert melody events to parallel token + beat-position arrays.

    Returns
    -------
    melody_tokens : int64 array [seq_len]
    beat_positions : int64 array [seq_len]  (position within bar, 0..15)
    """
    tokens = np.array(
        [melody_event_to_token(e) for e in events], dtype=np.int64,
    )
    positions = np.array(
        [e.position % MAX_BAR_POSITIONS for e in events], dtype=np.int64,
    )
    return tokens, positions


# ---- padding helper ------------------------------------------------------

def pad_sequence(arr: np.ndarray, max_len: int, pad_value: int) -> np.ndarray:
    if len(arr) >= max_len:
        return arr[:max_len].copy()
    out = np.full(max_len, pad_value, dtype=arr.dtype)
    out[: len(arr)] = arr
    return out
