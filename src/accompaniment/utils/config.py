"""Configuration loading and defaults."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "data": {
        "grid_resolution": 16,
        "chord_per": "beat",
        "max_seq_len": 256,
        "train_split": 0.9,
        "data_dir": "data/processed",
        "num_examples": 500,
    },
    "model": {
        "melody_vocab_size": 15,
        "chord_vocab_size": 86,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "max_seq_len": 256,
        "num_bar_positions": 16,
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 50,
        "patience": 10,
        "checkpoint_dir": "outputs/checkpoints",
        "log_dir": "outputs/logs",
    },
    "generation": {
        "default_pattern": "bass_chord",
        "accomp_octave": 3,
        "velocity": 70,
    },
    "export": {
        "opset_version": 14,
        "output_dir": "exports",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = DEFAULTS.copy()
    cfg = _deep_merge(cfg, DEFAULTS)
    if path is not None:
        file_cfg = load_config(path)
        cfg = _deep_merge(cfg, file_cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
