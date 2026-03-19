"""Deterministic accompaniment pattern library.

Patterns are functions that take a chord spelling + time position and
return a list of ``AccompNote`` objects ready for MIDI rendering.
"""

from __future__ import annotations

from typing import NamedTuple

from ..io.chord_vocab import chord_to_pitches, decode_chord, NO_CHORD_ID, PAD_CHORD_ID


class AccompNote(NamedTuple):
    pitch: int
    start_beat: float
    duration: float      # in beats
    velocity: int


# ---------------------------------------------------------------------------
# Voice-leading helper
# ---------------------------------------------------------------------------

def _voice_lead(prev: list[int], target: list[int]) -> list[int]:
    """Minimal-motion voice leading from *prev* to *target*."""
    if not prev or not target:
        return target
    ref = sum(prev) / len(prev)
    result: list[int] = []
    for t in target:
        candidates = [t + o * 12 for o in range(-2, 3)]
        best = min(candidates, key=lambda p: abs(p - ref))
        result.append(max(36, min(72, best)))
    return sorted(result)


# ---------------------------------------------------------------------------
# Pattern implementations
# ---------------------------------------------------------------------------

def pattern_block_chords(
    root: str, quality: str, beat_offset: float, beats: int,
    octave: int = 3, velocity: int = 70,
    prev_pitches: list[int] | None = None,
) -> tuple[list[AccompNote], list[int]]:
    pitches = chord_to_pitches(root, quality, octave)
    if prev_pitches:
        pitches = _voice_lead(prev_pitches, pitches)
    notes = [AccompNote(p, beat_offset, float(beats), velocity) for p in pitches]
    return notes, pitches


def pattern_half_note_shells(
    root: str, quality: str, beat_offset: float, beats: int,
    octave: int = 3, velocity: int = 65,
    prev_pitches: list[int] | None = None,
) -> tuple[list[AccompNote], list[int]]:
    pitches = chord_to_pitches(root, quality, octave)
    if prev_pitches:
        pitches = _voice_lead(prev_pitches, pitches)
    if not pitches:
        return [], []
    half = beats / 2.0
    notes: list[AccompNote] = []
    notes.append(AccompNote(pitches[0], beat_offset, half, velocity))
    if len(pitches) > 2:
        notes.append(AccompNote(pitches[-1], beat_offset, half, velocity))
    notes.append(AccompNote(pitches[0], beat_offset + half, half, velocity - 5))
    if len(pitches) > 1:
        notes.append(AccompNote(pitches[1], beat_offset + half, half, velocity - 5))
    return notes, pitches


def pattern_quarter_arpeggio(
    root: str, quality: str, beat_offset: float, beats: int,
    octave: int = 3, velocity: int = 60,
    prev_pitches: list[int] | None = None,
) -> tuple[list[AccompNote], list[int]]:
    pitches = chord_to_pitches(root, quality, octave)
    if prev_pitches:
        pitches = _voice_lead(prev_pitches, pitches)
    if not pitches:
        return [], []
    notes = [
        AccompNote(pitches[i % len(pitches)], beat_offset + i, 0.9, velocity)
        for i in range(beats)
    ]
    return notes, pitches


def pattern_eighth_broken(
    root: str, quality: str, beat_offset: float, beats: int,
    octave: int = 3, velocity: int = 55,
    prev_pitches: list[int] | None = None,
) -> tuple[list[AccompNote], list[int]]:
    pitches = chord_to_pitches(root, quality, octave)
    if prev_pitches:
        pitches = _voice_lead(prev_pitches, pitches)
    if not pitches:
        return [], []
    notes = [
        AccompNote(
            pitches[i % len(pitches)], beat_offset + i * 0.5, 0.45, velocity,
        )
        for i in range(beats * 2)
    ]
    return notes, pitches


def pattern_bass_chord(
    root: str, quality: str, beat_offset: float, beats: int,
    octave: int = 3, velocity: int = 70,
    prev_pitches: list[int] | None = None,
) -> tuple[list[AccompNote], list[int]]:
    pitches = chord_to_pitches(root, quality, octave)
    if prev_pitches:
        pitches = _voice_lead(prev_pitches, pitches)
    if not pitches:
        return [], []
    bass = min(pitches)
    upper = [p for p in pitches if p != bass] or pitches
    notes: list[AccompNote] = []
    for i in range(beats):
        if i % 2 == 0:
            notes.append(AccompNote(bass, beat_offset + i, 0.9, velocity))
        else:
            for p in upper:
                notes.append(AccompNote(p, beat_offset + i, 0.9, velocity - 10))
    return notes, pitches


# ---------------------------------------------------------------------------
# Pattern registry and selector
# ---------------------------------------------------------------------------

PATTERNS = {
    "block": pattern_block_chords,
    "shell": pattern_half_note_shells,
    "arpeggio": pattern_quarter_arpeggio,
    "broken": pattern_eighth_broken,
    "bass_chord": pattern_bass_chord,
}


def select_pattern(
    tempo: float = 120.0,
    note_density: float = 0.5,
    time_sig: tuple[int, int] = (4, 4),
    pattern_name: str | None = None,
) -> str:
    if pattern_name and pattern_name in PATTERNS:
        return pattern_name
    if tempo > 160:
        return "block"
    if tempo < 80:
        return "broken"
    if note_density > 0.7:
        return "shell"
    if note_density < 0.3:
        return "arpeggio"
    return "bass_chord"


# ---------------------------------------------------------------------------
# Top-level accompaniment generator
# ---------------------------------------------------------------------------


def generate_accompaniment(
    chord_ids: list[int],
    tempo: float = 120.0,
    beats_per_chord: int = 1,
    octave: int = 3,
    velocity: int = 70,
    pattern_name: str | None = None,
    time_sig: tuple[int, int] = (4, 4),
    note_density: float = 0.5,
) -> list[AccompNote]:
    pattern_key = select_pattern(tempo, note_density, time_sig, pattern_name)
    pattern_fn = PATTERNS[pattern_key]

    all_notes: list[AccompNote] = []
    prev_pitches: list[int] | None = None
    beat = 0.0
    i = 0

    while i < len(chord_ids):
        cid = chord_ids[i]
        # Count run of identical consecutive chords
        run = 1
        while i + run < len(chord_ids) and chord_ids[i + run] == cid:
            run += 1

        root, quality = decode_chord(cid)
        total_beats = run * beats_per_chord

        if cid not in (NO_CHORD_ID, PAD_CHORD_ID) and root != "N.C.":
            notes, prev_pitches = pattern_fn(
                root, quality, beat, total_beats,
                octave=octave, velocity=velocity,
                prev_pitches=prev_pitches,
            )
            all_notes.extend(notes)

        beat += total_beats
        i += run

    return all_notes
