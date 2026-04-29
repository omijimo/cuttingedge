#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/small.yaml}"
CHECKPOINT_DIR="${2:-outputs/checkpoints}"
OUTDIR="${3:-exports}"

echo "=== Preprocessing dataset ==="
python -m accompaniment.cli preprocess --config "$CONFIG"

echo "=== Training model ==="
python -m accompaniment.cli train --config "$CONFIG"

CHECKPOINT="$CHECKPOINT_DIR/best.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Expected checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "=== Exporting ONNX / Core ML / TFLite ==="
python -m accompaniment.cli export \
  --checkpoint "$CHECKPOINT" \
  --outdir "$OUTDIR"

echo "=== Done ==="
echo "Checkpoint: $CHECKPOINT"
echo "Exports:"
echo "  $OUTDIR/chord_model.onnx"
echo "  $OUTDIR/chord_model.tflite"
