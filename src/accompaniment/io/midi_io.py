"""MIDI file reading and writing via pretty_midi."""

from __future__ import annotations

from pathlib import Path
import pretty_midi


def load_midi(path: str | Path) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(path))


def save_midi(pm: pretty_midi.PrettyMIDI, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def get_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    tempos = pm.get_tempo_changes()[1]
    return float(tempos[0]) if len(tempos) > 0 else 120.0


def get_time_signature(pm: pretty_midi.PrettyMIDI) -> tuple[int, int]:
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        return (ts.numerator, ts.denominator)
    return (4, 4)


def get_total_beats(pm: pretty_midi.PrettyMIDI) -> int:
    return len(pm.get_beats())


def create_output_midi(
    melody_notes: list[tuple[int, float, float, int]],
    accomp_notes: list[tuple[int, float, float, int]],
    tempo: float = 120.0,
) -> pretty_midi.PrettyMIDI:
    """Build a PrettyMIDI with melody (track 0) and accompaniment (track 1).

    Each note is ``(pitch, start_time, end_time, velocity)``.
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    melody_inst = pretty_midi.Instrument(program=0, name="Melody")
    for pitch, start, end, vel in melody_notes:
        melody_inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=int(pitch), start=start, end=end)
        )
    pm.instruments.append(melody_inst)

    accomp_inst = pretty_midi.Instrument(program=0, name="Accompaniment")
    for pitch, start, end, vel in accomp_notes:
        accomp_inst.notes.append(
            pretty_midi.Note(velocity=vel, pitch=int(pitch), start=start, end=end)
        )
    pm.instruments.append(accomp_inst)

    return pm
