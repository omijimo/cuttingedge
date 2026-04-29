"""Render melody events + accompaniment notes into a PrettyMIDI object."""

from __future__ import annotations

from pathlib import Path
import pretty_midi

from .accompaniment_rules import AccompNote
from ..io.midi_io import create_output_midi, save_midi
from ..io.tokenization import MelodyEvent


def accomp_notes_to_tuples(
    notes: list[AccompNote], tempo: float = 120.0,
) -> list[tuple[int, float, float, int]]:
    """(pitch, start_sec, end_sec, velocity)"""
    beat_dur = 60.0 / tempo
    return [
        (n.pitch, n.start_beat * beat_dur, (n.start_beat + n.duration) * beat_dur, n.velocity)
        for n in notes
    ]


def melody_events_to_tuples(
    events: list[MelodyEvent],
    tempo: float = 120.0,
    grid_resolution: int = 16,
    default_velocity: int = 90,
) -> list[tuple[int, float, float, int]]:
    beat_dur = 60.0 / tempo
    steps_per_beat = grid_resolution // 4
    step_dur = beat_dur / steps_per_beat

    notes: list[tuple[int, float, float, int]] = []
    cur_pitch = -1
    note_start = 0.0

    for i, ev in enumerate(events):
        t = i * step_dur
        if ev.state == "on":
            if cur_pitch > 0:
                notes.append((cur_pitch, note_start, t, default_velocity))
            cur_pitch = ev.pitch
            note_start = t
        elif ev.state == "rest":
            if cur_pitch > 0:
                notes.append((cur_pitch, note_start, t, default_velocity))
            cur_pitch = -1

    if cur_pitch > 0:
        end_t = len(events) * step_dur
        notes.append((cur_pitch, note_start, end_t, default_velocity))

    return notes


def render_output(
    melody_events: list[MelodyEvent],
    accomp_notes: list[AccompNote],
    tempo: float = 120.0,
    grid_resolution: int = 16,
    output_path: str | Path | None = None,
) -> pretty_midi.PrettyMIDI:
    mel_tuples = melody_events_to_tuples(melody_events, tempo, grid_resolution)
    acc_tuples = accomp_notes_to_tuples(accomp_notes, tempo)

    # Clip accompaniment so it does not continue after melody ends
    if mel_tuples:
        melody_end = max(end for _, _, end, _ in mel_tuples)

        clipped_acc = []
        for pitch, start, end, velocity in acc_tuples:
            if start >= melody_end:
                continue
            end = min(end, melody_end)
            clipped_acc.append((pitch, start, end, velocity))

        acc_tuples = clipped_acc

    pm = create_output_midi(mel_tuples, acc_tuples, tempo)
    if output_path:
        save_midi(pm, output_path)
    return pm