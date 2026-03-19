"""Preprocessing pipeline: synthetic data generator + MIDI adapter."""

from __future__ import annotations

import json
import random
import numpy as np
from pathlib import Path
from typing import Iterator

from ..io.midi_io import load_midi, get_tempo, get_time_signature
from ..io.melody_extract import select_melody_track, extract_melody_events
from ..io.tokenization import MelodyEvent, tokenize_melody
from ..io.chord_vocab import (
    ROOTS,
    QUALITIES,
    encode_chord,
    parse_chord_label,
    NO_CHORD_ID,
)

# ---------------------------------------------------------------------------
# Common chord progressions (root, quality) in C; transposed at runtime
# ---------------------------------------------------------------------------

PROGRESSIONS: list[list[tuple[str, str]]] = [
    [("C", "maj"), ("G", "maj"), ("A", "min"), ("F", "maj")],       # I-V-vi-IV
    [("C", "maj"), ("F", "maj"), ("G", "maj"), ("C", "maj")],       # I-IV-V-I
    [("D", "min"), ("G", "dom7"), ("C", "maj7"), ("C", "maj7")],    # ii-V-I
    [("C", "maj"), ("A", "min"), ("F", "maj"), ("G", "maj")],       # I-vi-IV-V
    [("A", "min"), ("F", "maj"), ("C", "maj"), ("G", "maj")],       # vi-IV-I-V
    [("C", "maj"), ("F", "maj"), ("A", "min"), ("G", "maj")],       # I-IV-vi-V
    [("G", "maj"), ("D", "maj"), ("C", "maj"), ("G", "maj")],       # I-V-IV-I (G)
    [("C", "min"), ("F", "dom7"), ("A#", "maj7"), ("A#", "maj7")],  # ii-V-I (Bb)
    [("C", "maj"), ("E", "min"), ("A", "min"), ("G", "maj")],       # I-iii-vi-V
    [("C", "maj"), ("G", "maj"), ("A", "min"), ("E", "min")],       # I-V-vi-iii
]


def _transpose(prog: list[tuple[str, str]], semitones: int) -> list[tuple[str, str]]:
    out = []
    for root, quality in prog:
        idx = ROOTS.index(root)
        out.append((ROOTS[(idx + semitones) % 12], quality))
    return out


_CHORD_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6],
    "aug": [0, 4, 8], "dom7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}


def _chord_pcs(root: str, quality: str) -> list[int]:
    base = ROOTS.index(root)
    return [(base + i) % 12 for i in _CHORD_INTERVALS.get(quality, [0, 4, 7])]


# ---------------------------------------------------------------------------
# Synthetic melody generator
# ---------------------------------------------------------------------------


def generate_synthetic_melody(
    chords: list[tuple[str, str]],
    beats_per_chord: int = 4,
    grid_resolution: int = 16,
    rng: random.Random | None = None,
) -> list[MelodyEvent]:
    if rng is None:
        rng = random.Random(42)

    events: list[MelodyEvent] = []
    steps_per_beat = grid_resolution // 4
    octave = 5
    prev_pitch = -1

    for chord_idx, (root, quality) in enumerate(chords):
        pcs = _chord_pcs(root, quality)

        for beat in range(beats_per_chord):
            beat_abs = chord_idx * beats_per_chord + beat
            beat_in_bar = beat_abs % 4
            global_bar = beat_abs // 4

            for step in range(steps_per_beat):
                pos = beat_in_bar * steps_per_beat + step

                if step == 0:
                    if rng.random() < 0.85:
                        pc = rng.choice(pcs) if rng.random() < 0.6 else rng.randint(0, 11)
                        pitch = pc + octave * 12
                        if prev_pitch > 0:
                            while abs(pitch - prev_pitch) > 7 and pitch > 48:
                                pitch -= 12
                            while abs(pitch - prev_pitch) > 7 and pitch < 84:
                                pitch += 12
                        pitch = max(48, min(84, pitch))
                        prev_pitch = pitch
                        events.append(MelodyEvent(pitch, "on", global_bar, pos))
                    else:
                        events.append(MelodyEvent(-1, "rest", global_bar, pos))
                elif step == 1 and rng.random() < 0.35:
                    pc = rng.choice(pcs)
                    pitch = pc + octave * 12
                    pitch = max(48, min(84, pitch))
                    prev_pitch = pitch
                    events.append(MelodyEvent(pitch, "on", global_bar, pos))
                else:
                    if prev_pitch > 0 and events and events[-1].state != "rest":
                        events.append(MelodyEvent(prev_pitch, "hold", global_bar, pos))
                    else:
                        events.append(MelodyEvent(-1, "rest", global_bar, pos))

    return events


# ---------------------------------------------------------------------------
# Synthetic dataset generator
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_examples: int = 200,
    num_bars: int = 8,
    grid_resolution: int = 16,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_idx: list[dict] = []
    val_idx: list[dict] = []

    for i in range(num_examples):
        prog = rng.choice(PROGRESSIONS)
        prog = _transpose(prog, rng.randint(0, 11))

        repeats = max(1, num_bars // len(prog))
        full_chords = prog * repeats

        beats_per_chord = 4
        melody = generate_synthetic_melody(
            full_chords, beats_per_chord, grid_resolution, rng,
        )

        mel_tokens, beat_pos = tokenize_melody(melody)

        # Build timestep-level chord targets (repeat each beat chord across its steps)
        steps_per_beat = grid_resolution // 4
        chord_targets: list[int] = []
        for root, quality in full_chords:
            cid = encode_chord(root, quality)
            chord_targets.extend([cid] * (beats_per_chord * steps_per_beat))

        min_len = min(len(mel_tokens), len(chord_targets))
        mel_tokens = mel_tokens[:min_len]
        beat_pos = beat_pos[:min_len]
        ct = np.array(chord_targets[:min_len], dtype=np.int64)

        is_val = i >= int(num_examples * 0.9)
        dest = val_dir if is_val else train_dir
        fname = f"example_{i:05d}.npz"
        fpath = dest / fname

        np.savez(fpath, melody_tokens=mel_tokens, beat_positions=beat_pos, chord_targets=ct)

        entry = {"path": str(fpath)}
        (val_idx if is_val else train_idx).append(entry)

    with open(output_dir / "train.json", "w") as f:
        json.dump(train_idx, f, indent=2)
    with open(output_dir / "val.json", "w") as f:
        json.dump(val_idx, f, indent=2)


# ---------------------------------------------------------------------------
# MIDI + chord-annotation adapter (for real datasets)
# ---------------------------------------------------------------------------


class MidiChordAdapter:
    """Process MIDI files with optional text-based chord annotation files.

    Chord file format: one line per chord change, ``<time_seconds> <label>``.
    """

    def __init__(
        self,
        midi_dir: str | Path,
        chord_dir: str | Path | None = None,
        grid_resolution: int = 16,
    ):
        self.midi_dir = Path(midi_dir)
        self.chord_dir = Path(chord_dir) if chord_dir else None
        self.grid_resolution = grid_resolution

    def _parse_chord_file(
        self, path: Path, num_steps: int, tempo: float,
    ) -> np.ndarray | None:
        chords: list[tuple[float, int]] = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    t = float(parts[0])
                    root, quality = parse_chord_label(parts[1])
                    chords.append((t, encode_chord(root, quality)))
        if not chords:
            return None

        beat_dur = 60.0 / tempo
        step_dur = beat_dur / (self.grid_resolution // 4)
        targets = np.full(num_steps, NO_CHORD_ID, dtype=np.int64)
        for i in range(num_steps):
            t = i * step_dur
            cur = chords[0][1]
            for ct, cid in chords:
                if ct <= t:
                    cur = cid
                else:
                    break
            targets[i] = cur
        return targets

    def process_file(
        self, midi_path: Path, chord_path: Path | None = None,
    ) -> dict | None:
        try:
            pm = load_midi(midi_path)
            tempo = get_tempo(pm)
            time_sig = get_time_signature(pm)
            inst = select_melody_track(pm)
            events = extract_melody_events(inst, tempo, self.grid_resolution, time_sig)
            if len(events) < 16:
                return None
            mel_tokens, beat_pos = tokenize_melody(events)
            chord_targets = None
            if chord_path and chord_path.exists():
                chord_targets = self._parse_chord_file(chord_path, len(mel_tokens), tempo)
            return {
                "melody_tokens": mel_tokens,
                "beat_positions": beat_pos,
                "chord_targets": chord_targets,
                "metadata": {"tempo": tempo, "time_sig": time_sig, "source": str(midi_path)},
            }
        except Exception:
            return None

    def iter_examples(self) -> Iterator[dict]:
        for midi_path in sorted(self.midi_dir.glob("**/*.mid")):
            chord_path = None
            if self.chord_dir:
                chord_path = self.chord_dir / midi_path.with_suffix(".txt").name
            result = self.process_file(midi_path, chord_path)
            if result is not None:
                yield result
