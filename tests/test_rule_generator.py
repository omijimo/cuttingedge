"""Tests for the deterministic accompaniment generator."""

from accompaniment.io.chord_vocab import encode_chord
from accompaniment.generation.accompaniment_rules import (
    generate_accompaniment,
    pattern_block_chords,
    pattern_quarter_arpeggio,
    pattern_bass_chord,
    PATTERNS,
)


def test_block_chords_basic():
    notes, pitches = pattern_block_chords("C", "maj", 0.0, 4, octave=3)
    assert len(notes) == 3  # root + third + fifth
    assert all(n.start_beat == 0.0 for n in notes)
    assert all(n.duration == 4.0 for n in notes)


def test_arpeggio_basic():
    notes, pitches = pattern_quarter_arpeggio("G", "min", 0.0, 4, octave=3)
    assert len(notes) == 4
    for i, n in enumerate(notes):
        assert n.start_beat == float(i)
        assert abs(n.duration - 0.9) < 0.01


def test_bass_chord_pattern():
    notes, pitches = pattern_bass_chord("C", "maj", 0.0, 4, octave=3)
    assert len(notes) > 0
    # Strong beats get bass, weak beats get upper
    assert notes[0].pitch == min(pitches)


def test_all_patterns_registered():
    expected = {"block", "shell", "arpeggio", "broken", "bass_chord"}
    assert set(PATTERNS.keys()) == expected


def test_generate_accompaniment_basic():
    chord_ids = [
        encode_chord("C", "maj"),
        encode_chord("C", "maj"),
        encode_chord("G", "maj"),
        encode_chord("G", "maj"),
    ]
    notes = generate_accompaniment(chord_ids, tempo=120.0, pattern_name="block")
    assert len(notes) > 0
    # All notes should have valid MIDI pitches
    for n in notes:
        assert 0 <= n.pitch <= 127


def test_generate_accompaniment_with_no_chord():
    from accompaniment.io.chord_vocab import NO_CHORD_ID

    chord_ids = [encode_chord("C", "maj"), NO_CHORD_ID, encode_chord("G", "maj")]
    notes = generate_accompaniment(chord_ids, tempo=120.0, pattern_name="arpeggio")
    # Should not crash; no-chord beats produce no notes
    assert isinstance(notes, list)
