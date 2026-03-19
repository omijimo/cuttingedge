"""Tests for MIDI I/O utilities."""

from pathlib import Path

import pretty_midi

from accompaniment.io.midi_io import (
    load_midi,
    save_midi,
    get_tempo,
    get_time_signature,
    create_output_midi,
)


def test_load_and_save_roundtrip(tmp_midi: Path, tmp_path: Path):
    pm = load_midi(tmp_midi)
    assert len(pm.instruments) == 1
    assert len(pm.instruments[0].notes) == 8

    out = tmp_path / "round.mid"
    save_midi(pm, out)
    pm2 = load_midi(out)
    assert len(pm2.instruments[0].notes) == 8


def test_get_tempo(tmp_midi: Path):
    pm = load_midi(tmp_midi)
    assert abs(get_tempo(pm) - 120.0) < 1.0


def test_get_time_signature(tmp_midi: Path):
    pm = load_midi(tmp_midi)
    assert get_time_signature(pm) == (4, 4)


def test_create_output_midi():
    melody = [(60, 0.0, 0.5, 80), (62, 0.5, 1.0, 80)]
    accomp = [(48, 0.0, 1.0, 70)]
    pm = create_output_midi(melody, accomp, tempo=120.0)
    assert len(pm.instruments) == 2
    assert pm.instruments[0].name == "Melody"
    assert pm.instruments[1].name == "Accompaniment"
