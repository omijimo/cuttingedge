from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np

from accompaniment.data.preprocess import MidiChordAdapter


def find_midi_file(song_dir: Path) -> Path | None:
    song_id = song_dir.name

    candidates = [
        song_dir / f"{song_id}.mid",
        song_dir / song_id,
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    mids = sorted(song_dir.glob("*.mid"))
    if mids:
        return mids[0]

    return None


def find_chord_audio_file(song_dir: Path) -> Path | None:
    candidates = [
        song_dir / "chord_audio.txt",
        song_dir / "chord_audio",
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def convert_pop909_chord_line(line: str) -> str | None:
    """
    Convert a POP909 chord line into repo format: '<time_seconds> <chord_label>'.

    Expected common POP909-style cases:
    - '0.000 2.000 C:maj'
    - '0.000 C:maj'
    """
    parts = line.strip().split()
    if not parts:
        return None

    if len(parts) >= 3:
        start = parts[0]
        chord = parts[2]
        return f"{start} {chord}"

    if len(parts) == 2:
        start = parts[0]
        chord = parts[1]
        return f"{start} {chord}"

    return None


def normalize_chord_file(src: Path, dst: Path) -> int:
    kept = 0
    out_lines: list[str] = []

    with open(src, "r", encoding="utf-8") as f:
        for raw in f:
            converted = convert_pop909_chord_line(raw)
            if converted is not None:
                out_lines.append(converted + "\n")
                kept += 1

    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return kept


def save_examples(
    examples: list[dict],
    split_dir: Path,
    manifest_path: Path,
    prefix: str,
) -> int:
    manifest: list[dict] = []
    count = 0

    split_dir.mkdir(parents=True, exist_ok=True)

    for i, ex in enumerate(examples):
        # Skip bad or empty examples
        if ex is None:
            continue
        if ex.get("chord_targets") is None:
            continue

        melody_tokens = ex["melody_tokens"]
        beat_positions = ex["beat_positions"]
        chord_targets = ex["chord_targets"]

        out_path = split_dir / f"{prefix}_{i:05d}.npz"
        np.savez(
            out_path,
            melody_tokens=melody_tokens,
            beat_positions=beat_positions,
            chord_targets=chord_targets,
        )
        manifest.append({"path": str(out_path)})
        count += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw POP909 folder containing subfolders like 001, 002, ...",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="data",
        help="Working directory where converted files and processed dataset will be written.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_resolution", type=int, default=16)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    work_dir = Path(args.work_dir)

    midi_out_dir = work_dir / "pop909_midis"
    chord_out_dir = work_dir / "pop909_chords"
    processed_dir = work_dir / "processed"
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"

    midi_out_dir.mkdir(parents=True, exist_ok=True)
    chord_out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    copied_midis = 0
    converted_chords = 0
    skipped = []

    song_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])

    for song_dir in song_dirs:
        song_id = song_dir.name

        midi_src = find_midi_file(song_dir)
        chord_src = find_chord_audio_file(song_dir)

        if midi_src is None or chord_src is None:
            skipped.append(song_id)
            continue

        midi_dst = midi_out_dir / f"{song_id}.mid"
        chord_dst = chord_out_dir / f"{song_id}.txt"

        shutil.copy2(midi_src, midi_dst)
        copied_midis += 1

        kept = normalize_chord_file(chord_src, chord_dst)
        if kept > 0:
            converted_chords += 1
        else:
            skipped.append(song_id)

    print(f"Copied MIDI files: {copied_midis}")
    print(f"Converted chord files: {converted_chords}")
    if skipped:
        print(f"Skipped songs: {len(skipped)}")
        print(", ".join(skipped[:20]))

    adapter = MidiChordAdapter(
        midi_dir=midi_out_dir,
        chord_dir=chord_out_dir,
        grid_resolution=args.grid_resolution,
    )

    examples = list(adapter.iter_examples())
    print(f"Adapter yielded {len(examples)} examples")

    if not examples:
        raise RuntimeError(
            "No examples were produced. Check your raw_dir path and chord file format."
        )

    random.Random(args.seed).shuffle(examples)

    split_idx = int(len(examples) * args.train_ratio)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    train_count = save_examples(
        train_examples,
        train_dir,
        processed_dir / "train.json",
        "train",
    )
    val_count = save_examples(
        val_examples,
        val_dir,
        processed_dir / "val.json",
        "val",
    )

    print("Done.")
    print(f"Saved train examples: {train_count}")
    print(f"Saved val examples: {val_count}")
    print(f"Processed dataset directory: {processed_dir}")


if __name__ == "__main__":
    main()