# Accompaniment

Symbolic music accompaniment generation: a compact melody-to-chords transformer
followed by a deterministic accompaniment pattern generator.
Designed for fast iteration, local training, and edge-device export.

## Architecture

```
┌─────────────┐      ┌───────────────────────┐      ┌────────────────────────┐
│  Input MIDI  │─────▶│  Melody Extraction &  │─────▶│  Stage A: Chord        │
│  (melody)    │      │  Tokenization         │      │  Transformer           │
└─────────────┘      └───────────────────────┘      │  (learned, PyTorch)    │
                                                      └──────────┬─────────────┘
                                                                 │ chord IDs
                                                                 ▼
                     ┌───────────────────────┐      ┌────────────────────────┐
                     │  Output MIDI          │◀─────│  Stage B: Deterministic│
                     │  (melody + accomp)    │      │  Accompaniment Rules   │
                     └───────────────────────┘      └────────────────────────┘
```

**Stage A** is a small encoder transformer (~50-200K params) that reads a
tokenised melody sequence and predicts one chord label per timestep.
At beat boundaries, predictions are aggregated by majority vote.

**Stage B** takes the predicted chord sequence and the original melody,
then applies configurable accompaniment patterns (block chords, arpeggios,
bass + chord comping, etc.) to produce a piano accompaniment MIDI track.

## Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,export]"

# 2. Generate synthetic training data
python -m accompaniment.cli preprocess --config configs/base.yaml

# 3. Train (small config, ~1 min on CPU)
python -m accompaniment.cli train --config configs/small.yaml

# 4. Run inference on a MIDI file
python -m accompaniment.cli infer \
    --checkpoint outputs/checkpoints/best.pt \
    --input data/sample_input/demo.mid \
    --output data/sample_output/demo_accomp.mid

# 5. Export to ONNX
python -m accompaniment.cli export \
    --checkpoint outputs/checkpoints/best.pt \
    --outdir exports/

# 6. Run tests
pytest tests/ -v
```

## Project Structure

```
├── configs/                 # YAML configuration files
│   ├── base.yaml            # Default / preprocessing config
│   ├── small.yaml           # Fast smoke-test config
│   └── export.yaml          # Export-specific overrides
├── data/
│   ├── sample_input/        # Input MIDI files
│   ├── sample_output/       # Generated output MIDI files
│   └── processed/           # Preprocessed dataset (git-ignored)
├── scripts/                 # Convenience shell scripts
├── src/accompaniment/
│   ├── cli.py               # CLI entry point
│   ├── io/                  # MIDI I/O, tokenization, chord vocabulary
│   ├── data/                # Dataset, preprocessing, collation
│   ├── models/              # Chord transformer model
│   ├── generation/          # Chord decoding + accompaniment rules
│   ├── training/            # Training loop, evaluation, metrics
│   ├── export/              # ONNX, Core ML, TFLite export
│   └── utils/               # Config, logging, seeding
└── tests/                   # pytest test suite
```

## Dataset Preparation

### Synthetic Data (default)

```bash
python -m accompaniment.cli preprocess --config configs/base.yaml
```

Generates melody–chord pairs from common progressions (I-V-vi-IV, ii-V-I, etc.)
transposed to all keys. Good enough for testing the full pipeline.

### Real Datasets

The preprocessing pipeline includes a `MidiChordAdapter` for loading MIDI files
with text-based chord annotations. See `data/README.md` for format details.

**Adapter priority:**
1. POP909-style adapter (if files available)
2. Generic MIDI + chord annotation adapter
3. Synthetic fallback (always works)

## Configuration

All settings live in YAML config files. Key parameters:

| Section    | Parameter          | Description                        |
|------------|--------------------|------------------------------------|
| `data`     | `grid_resolution`  | Quantization grid (default: 16th)  |
| `data`     | `max_seq_len`      | Max tokens per example             |
| `model`    | `d_model`          | Hidden dimension (64–256)          |
| `model`    | `num_layers`       | Transformer layers (1–4)           |
| `model`    | `nhead`            | Attention heads                    |
| `training` | `batch_size`       | Training batch size                |
| `training` | `epochs`           | Max epochs                         |
| `training` | `patience`         | Early stopping patience            |

## Model Details

### Chord Vocabulary

12 roots × 7 qualities + no-chord + pad = **86 tokens**

Qualities: `maj`, `min`, `dim`, `aug`, `dom7`, `maj7`, `min7`

### Melody Tokens

15-token vocabulary: 12 pitch classes + HOLD + REST + PAD

### Accompaniment Patterns

| Pattern       | Description                              |
|---------------|------------------------------------------|
| `block`       | Whole-note block chords                  |
| `shell`       | Half-note shell voicings (root + 3rd/7th)|
| `arpeggio`    | Quarter-note arpeggiation                |
| `broken`      | Eighth-note broken chord pattern         |
| `bass_chord`  | Bass on strong beats, chord on weak beats|

Pattern selection is automatic based on tempo and density, or can be
overridden via the `generation.default_pattern` config.

## Export

### ONNX (recommended for edge)

```bash
python -m accompaniment.cli export --checkpoint outputs/checkpoints/best.pt
```

Produces `exports/chord_model.onnx` with dynamic batch/sequence axes.
Verify with:

```python
from accompaniment.export.verify_exports import verify_onnx
verify_onnx("outputs/checkpoints/best.pt", "exports/chord_model.onnx")
```

### Core ML

Requires `coremltools`. Produces `.mlpackage` for Apple devices.

### TFLite

Best-effort via ONNX → TF → TFLite chain. Requires `onnx-tf` and `tensorflow`.
May not work for all op configurations; ONNX Runtime on mobile is a more
reliable alternative.

## Evaluation Metrics

- **Chord accuracy**: exact match of predicted vs target chord
- **Root accuracy**: correct root note regardless of quality
- **Quality accuracy**: correct quality regardless of root

```bash
python -m accompaniment.cli eval --checkpoint outputs/checkpoints/best.pt
```

## Testing

```bash
pytest tests/ -v
```

Tests cover: MIDI parsing, tokenization, model forward shapes, accompaniment
generation, and a full end-to-end pipeline on synthetic data.

### Scripted usage examples

#### Using `scripts/train_and_export.sh`

This script runs the full training/export pipeline:
1. `preprocess` with the provided config
2. `train` with the same config
3. `export` from `<checkpoint_dir>/best.pt` to `<outdir>`

```bash
# usage
scripts/train_and_export.sh [config] [checkpoint_dir] [outdir]

# defaults
scripts/train_and_export.sh
# equivalent to:
# scripts/train_and_export.sh configs/small.yaml outputs/checkpoints exports
```

Run a full training + export pipeline (synthetic preprocess, train, export ONNX/TFLite):

```bash
scripts/train_and_export.sh
```

Override config/checkpoint/export locations:

```bash
scripts/train_and_export.sh configs/base.yaml outputs/checkpoints exports
```

Run a `.tflite` model on a MIDI file and render the full accompaniment output MIDI:

```bash
python3 scripts/infer_tflite_midi.py \
  --model exports/chord_model.tflite \
  --input data/sample_input/demo.mid \
  --output data/sample_output/demo_tflite_accomp.mid \
  --chords-out data/sample_output/demo_tflite_chords.json
```

### Full training → inference pipeline example

```bash
# 1) preprocess synthetic data
python -m accompaniment.cli preprocess --config configs/small.yaml

# 2) train model checkpoint
python -m accompaniment.cli train --config configs/small.yaml

# 3) export ONNX/TFLite
python -m accompaniment.cli export \
  --checkpoint outputs/checkpoints/best.pt \
  --outdir exports

# 4) run PyTorch checkpoint inference
python -m accompaniment.cli infer \
  --checkpoint outputs/checkpoints/best.pt \
  --input data/sample_input/demo.mid \
  --output data/sample_output/demo_accomp.mid

# 5) run TFLite inference pipeline (desktop)
python3 scripts/infer_tflite_midi.py \
  --model exports/chord_model.tflite \
  --input data/sample_input/demo.mid \
  --output data/sample_output/demo_tflite_accomp.mid
```
