#!/usr/bin/env python3
"""Run end-to-end MIDI accompaniment inference using a TFLite chord model.
1) Load MIDI
2) Extract melody events and tokenize
3) Run TFLite model
4) Majority-vote decode to one chord per beat
5) Generate deterministic accompaniment
6) Render output MIDI (melody + accompaniment)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make "accompaniment" importable when script runs from repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from accompaniment.generation.accompaniment_rules import generate_accompaniment
from accompaniment.generation.chord_decode import decode_to_labels
from accompaniment.generation.midi_render import render_output
from accompaniment.io.chord_vocab import PAD_CHORD_ID
from accompaniment.io.melody_extract import extract_melody_events, select_melody_track
from accompaniment.io.midi_io import get_tempo, get_time_signature, load_midi
from accompaniment.io.tokenization import PAD_TOKEN, pad_sequence, tokenize_melody


def _load_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-untyped]
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]
        except ImportError:
            try:
                import tensorflow as tf  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError(
                    "No TFLite interpreter found. Install one of:\n"
                    "  pip install ai-edge-litert\n"
                    "  pip install tflite-runtime\n"
                    "  pip install tensorflow"
                ) from exc
            Interpreter = tf.lite.Interpreter

    return Interpreter(model_path=str(model_path))


def _decode_logits_to_chords(logits: np.ndarray, actual_len: int, steps_per_beat: int) -> list[int]:
    """Argmax per step, then majority vote per beat."""
    # logits: [1, seq_len, vocab] or [seq_len, vocab]
    if logits.ndim == 3:
        logits = logits[0]

    preds = np.argmax(logits[:actual_len], axis=-1)
    chord_ids: list[int] = []
    for beat_start in range(0, actual_len, steps_per_beat):
        beat_end = min(beat_start + steps_per_beat, actual_len)
        beat_preds = preds[beat_start:beat_end]
        valid = beat_preds[beat_preds != PAD_CHORD_ID]
        if valid.size == 0:
            chord_ids.append(PAD_CHORD_ID)
            continue
        vals, counts = np.unique(valid, return_counts=True)
        chord_ids.append(int(vals[np.argmax(counts)]))
    return chord_ids


def run(args: argparse.Namespace) -> None:
    pm = load_midi(args.input)
    tempo = get_tempo(pm)
    time_sig = get_time_signature(pm)

    melody_inst = select_melody_track(pm, track_index=args.track)
    melody_events = extract_melody_events(
        melody_inst,
        tempo=tempo,
        grid_resolution=args.grid_resolution,
        time_sig=time_sig,
    )
    if not melody_events:
        raise RuntimeError("No melody events extracted from input MIDI.")

    mel_tokens, beat_positions = tokenize_melody(melody_events)
    interpreter = _load_interpreter(Path(args.model))
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    if len(inp) < 2:
        raise RuntimeError(f"Expected 2 model inputs, found {len(inp)}")

    seq_len = int(inp[0]["shape"][1])
    mel_padded = pad_sequence(mel_tokens, seq_len, PAD_TOKEN)
    bpos_padded = pad_sequence(beat_positions, seq_len, 0)
    actual_len = min(len(mel_tokens), seq_len)

    # Model inputs are int64 in this project, but cast dynamically to match model dtypes.
    mel_batch = np.expand_dims(mel_padded, axis=0).astype(inp[0]["dtype"])
    bpos_batch = np.expand_dims(bpos_padded, axis=0).astype(inp[1]["dtype"])

    interpreter.set_tensor(inp[0]["index"], mel_batch)
    interpreter.set_tensor(inp[1]["index"], bpos_batch)
    interpreter.invoke()
    logits = interpreter.get_tensor(out[0]["index"])

    steps_per_beat = args.grid_resolution // 4
    chord_ids = _decode_logits_to_chords(logits, actual_len, steps_per_beat)
    chord_labels = decode_to_labels(chord_ids)

    accomp_notes = generate_accompaniment(
        chord_ids=chord_ids,
        tempo=tempo,
        beats_per_chord=1,
        octave=args.accomp_octave,
        velocity=args.velocity,
        pattern_name=args.pattern,
        time_sig=time_sig,
    )

    render_output(
        melody_events=melody_events,
        accomp_notes=accomp_notes,
        tempo=tempo,
        grid_resolution=args.grid_resolution,
        output_path=args.output,
    )

    print(f"Wrote MIDI: {args.output}")
    print(f"Predicted chords (first 16): {chord_labels[:16]}")

    if args.chords_out:
        payload = {"chords": chord_labels, "tempo": tempo, "time_signature": list(time_sig)}
        with open(args.chords_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote chord JSON: {args.chords_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TFLite MIDI accompaniment inference")
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--input", required=True, help="Path to input melody MIDI")
    parser.add_argument("--output", required=True, help="Path to output MIDI")
    parser.add_argument("--track", type=int, default=None, help="Optional non-drum track index")
    parser.add_argument("--grid-resolution", type=int, default=16, help="Tokenization grid resolution")
    parser.add_argument("--pattern", default=None, help="Accompaniment pattern override")
    parser.add_argument("--accomp-octave", type=int, default=3, help="Accompaniment octave")
    parser.add_argument("--velocity", type=int, default=70, help="Accompaniment note velocity")
    parser.add_argument("--chords-out", default=None, help="Optional path to save predicted chords JSON")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
