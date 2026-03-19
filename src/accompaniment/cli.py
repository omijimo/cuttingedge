"""CLI entry point for the accompaniment MVP pipeline."""

from __future__ import annotations

import argparse
import json
import logging as _logging
import sys
from pathlib import Path

from .utils.config import get_config
from .utils.logging import setup_logging


def cmd_preprocess(args: argparse.Namespace) -> None:
    cfg = get_config(args.config)
    from .data.preprocess import generate_synthetic_dataset

    dcfg = cfg["data"]
    log = _logging.getLogger("accompaniment")
    log.info("Generating synthetic dataset → %s", dcfg["data_dir"])
    generate_synthetic_dataset(
        output_dir=dcfg["data_dir"],
        num_examples=dcfg.get("num_examples", 200),
        grid_resolution=dcfg["grid_resolution"],
        seed=cfg.get("seed", 42),
    )
    log.info("Done.")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = get_config(args.config)
    from .training.train import train

    log = _logging.getLogger("accompaniment")
    log.info("Starting training")
    best = train(cfg)
    log.info("Best checkpoint: %s", best)


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = get_config(args.config)
    from .training.eval import evaluate

    log = _logging.getLogger("accompaniment")
    ckpt = args.checkpoint or str(Path(cfg["training"]["checkpoint_dir"]) / "best.pt")
    log.info("Evaluating %s", ckpt)
    metrics = evaluate(cfg, ckpt)
    print(json.dumps(metrics, indent=2))


def cmd_infer(args: argparse.Namespace) -> None:
    import torch
    from .io.midi_io import load_midi, get_tempo, get_time_signature
    from .io.melody_extract import select_melody_track, extract_melody_events
    from .io.tokenization import tokenize_melody, pad_sequence, PAD_TOKEN
    from .training.eval import load_model_from_checkpoint
    from .generation.chord_decode import greedy_decode, decode_to_labels
    from .generation.accompaniment_rules import generate_accompaniment
    from .generation.midi_render import render_output

    log = _logging.getLogger("accompaniment")
    cfg = get_config(args.config) if args.config else get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_cfg = load_model_from_checkpoint(args.checkpoint, device)
    gen_cfg = cfg.get("generation", {})
    data_cfg = model_cfg.get("data", cfg.get("data", {}))
    grid_res = data_cfg.get("grid_resolution", 16)
    max_seq = model_cfg["model"]["max_seq_len"]

    pm = load_midi(args.input)
    tempo = get_tempo(pm)
    time_sig = get_time_signature(pm)
    melody_inst = select_melody_track(pm, track_index=args.track)
    events = extract_melody_events(melody_inst, tempo, grid_res, time_sig)
    log.info("Extracted %d melody events at %.0f BPM", len(events), tempo)

    mel_tokens, beat_pos = tokenize_melody(events)
    mel_padded = pad_sequence(mel_tokens, max_seq, PAD_TOKEN)
    bpos_padded = pad_sequence(beat_pos, max_seq, 0)

    mel_t = torch.from_numpy(mel_padded).unsqueeze(0).to(device)
    bpos_t = torch.from_numpy(bpos_padded).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mel_t, bpos_t)

    actual_len = min(len(mel_tokens), max_seq)
    logits_trimmed = logits[0, :actual_len]

    steps_per_beat = grid_res // 4
    chord_ids = greedy_decode(logits_trimmed.unsqueeze(0), grid_res, steps_per_beat)
    chord_labels = decode_to_labels(chord_ids)
    log.info("Predicted chords: %s …", chord_labels[:16])

    accomp_notes = generate_accompaniment(
        chord_ids,
        tempo=tempo,
        beats_per_chord=1,
        octave=gen_cfg.get("accomp_octave", 3),
        velocity=gen_cfg.get("velocity", 70),
        pattern_name=gen_cfg.get("default_pattern"),
        time_sig=time_sig,
    )

    output_path = args.output or "output.mid"
    render_output(events, accomp_notes, tempo, grid_res, output_path)
    log.info("Wrote %s", output_path)

    if args.chords_out:
        with open(args.chords_out, "w") as f:
            json.dump({"chords": chord_labels, "tempo": tempo}, f, indent=2)
        log.info("Saved chord predictions → %s", args.chords_out)


def cmd_export(args: argparse.Namespace) -> None:
    cfg = get_config(args.config) if args.config else get_config()
    outdir = Path(args.outdir or cfg.get("export", {}).get("output_dir", "exports"))
    outdir.mkdir(parents=True, exist_ok=True)
    log = _logging.getLogger("accompaniment")

    from .export.export_onnx import export_onnx
    from .export.export_coreml import export_coreml
    from .export.export_tflite import export_tflite

    log.info("Exporting ONNX …")
    try:
        export_onnx(args.checkpoint, outdir / "chord_model.onnx", cfg)
        log.info("  ONNX ✓")
    except Exception as e:
        log.error("  ONNX failed: %s", e)

    log.info("Exporting Core ML …")
    try:
        export_coreml(args.checkpoint, outdir / "chord_model.mlpackage", cfg)
        log.info("  Core ML ✓")
    except Exception as e:
        log.error("  Core ML failed: %s", e)

    log.info("Exporting TFLite …")
    try:
        export_tflite(args.checkpoint, outdir / "chord_model.tflite", cfg)
        log.info("  TFLite ✓")
    except Exception as e:
        log.error("  TFLite failed: %s", e)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="accompaniment",
        description="Symbolic music accompaniment generation MVP",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="Generate / preprocess dataset")
    p.add_argument("--config", default="configs/base.yaml")

    p = sub.add_parser("train", help="Train chord prediction model")
    p.add_argument("--config", default="configs/small.yaml")

    p = sub.add_parser("eval", help="Evaluate model")
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--checkpoint", default=None)

    p = sub.add_parser("infer", help="Run inference on a MIDI file")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="output.mid")
    p.add_argument("--config", default=None)
    p.add_argument("--track", type=int, default=None)
    p.add_argument("--chords-out", default=None)

    p = sub.add_parser("export", help="Export model to ONNX / Core ML / TFLite")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--outdir", default="exports")
    p.add_argument("--config", default=None)

    args = parser.parse_args()
    setup_logging()

    dispatch = {
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "eval": cmd_eval,
        "infer": cmd_infer,
        "export": cmd_export,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
