"""Extract a melody line from a MIDI file and quantise to a grid."""

from __future__ import annotations

import numpy as np
import pretty_midi

from .tokenization import MelodyEvent


def _monophonic_score(inst: pretty_midi.Instrument) -> float:
    """Higher score = more likely to be a melody track."""
    if inst.is_drum or len(inst.notes) == 0:
        return -1.0
    notes = sorted(inst.notes, key=lambda n: n.start)
    overlaps = sum(
        1 for i in range(1, len(notes)) if notes[i].start < notes[i - 1].end - 0.01
    )
    mono_ratio = 1.0 - overlaps / max(len(notes) - 1, 1)
    avg_pitch = float(np.mean([n.pitch for n in notes]))
    return mono_ratio * 0.7 + (avg_pitch / 127.0) * 0.3


def select_melody_track(
    pm: pretty_midi.PrettyMIDI,
    track_index: int | None = None,
) -> pretty_midi.Instrument:
    non_drum = [inst for inst in pm.instruments if not inst.is_drum]
    if not non_drum:
        raise ValueError("No non-drum tracks in MIDI file")
    if track_index is not None:
        if track_index < len(non_drum):
            return non_drum[track_index]
        raise ValueError(
            f"Track index {track_index} out of range ({len(non_drum)} non-drum tracks)"
        )
    scores = [_monophonic_score(inst) for inst in non_drum]
    return non_drum[int(np.argmax(scores))]


def extract_melody_events(
    instrument: pretty_midi.Instrument,
    tempo: float = 120.0,
    grid_resolution: int = 16,
    time_sig: tuple[int, int] = (4, 4),
    max_bars: int | None = None,
) -> list[MelodyEvent]:
    """Quantise notes onto a 16th-note grid and return MelodyEvent list."""
    beat_duration = 60.0 / tempo
    steps_per_beat = grid_resolution // time_sig[0]
    step_duration = beat_duration / steps_per_beat

    notes = sorted(instrument.notes, key=lambda n: (n.start, -n.pitch))
    if not notes:
        return []

    total_time = notes[-1].end
    if max_bars is not None:
        bar_duration = beat_duration * time_sig[0]
        total_time = min(total_time, max_bars * bar_duration)

    num_steps = int(np.ceil(total_time / step_duration))
    if num_steps == 0:
        return []

    pitch_grid = np.full(num_steps, -1, dtype=np.int32)
    state_grid: list[str] = ["rest"] * num_steps

    for note in notes:
        start_step = int(round(note.start / step_duration))
        end_step = int(round(note.end / step_duration))
        start_step = max(0, min(start_step, num_steps - 1))
        end_step = max(start_step + 1, min(end_step, num_steps))

        # Prefer higher pitch when two notes overlap on the same step
        if pitch_grid[start_step] == -1 or note.pitch > pitch_grid[start_step]:
            pitch_grid[start_step] = note.pitch
            state_grid[start_step] = "on"
        for s in range(start_step + 1, end_step):
            if pitch_grid[s] == -1:
                pitch_grid[s] = note.pitch
                state_grid[s] = "hold"

    events: list[MelodyEvent] = []
    for i in range(num_steps):
        events.append(
            MelodyEvent(
                pitch=int(pitch_grid[i]),
                state=state_grid[i],
                bar=i // grid_resolution,
                position=i % grid_resolution,
            )
        )
    return events
