"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pretty_midi

# Ensure the package is importable even without pip install -e .
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def tmp_midi(tmp_path: Path) -> Path:
    """Create a tiny MIDI file with a C-major-scale melody."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    t = 0.0
    for p in pitches:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=t, end=t + 0.5))
        t += 0.5
    pm.instruments.append(inst)
    path = tmp_path / "test_melody.mid"
    pm.write(str(path))
    return path


@pytest.fixture
def tiny_config(tmp_path: Path) -> dict:
    """Minimal config for fast unit tests."""
    return {
        "seed": 42,
        "data": {
            "grid_resolution": 16,
            "chord_per": "beat",
            "max_seq_len": 64,
            "train_split": 0.9,
            "data_dir": str(tmp_path / "processed"),
            "num_examples": 20,
        },
        "model": {
            "melody_vocab_size": 15,
            "chord_vocab_size": 86,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 1,
            "dim_feedforward": 64,
            "dropout": 0.0,
            "max_seq_len": 64,
            "num_bar_positions": 16,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.002,
            "weight_decay": 0.0,
            "epochs": 2,
            "patience": 5,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
        },
        "generation": {
            "default_pattern": "block",
            "accomp_octave": 3,
            "velocity": 70,
        },
        "export": {
            "opset_version": 14,
            "output_dir": str(tmp_path / "exports"),
        },
    }
