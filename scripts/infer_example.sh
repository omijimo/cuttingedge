#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/checkpoints/best.pt}"
INPUT="${2:-data/sample_input/demo.mid}"
OUTPUT="${3:-data/sample_output/demo_accomp.mid}"

echo "=== Running inference ==="
python -m accompaniment.cli infer \
    --checkpoint "$CHECKPOINT" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --chords-out "${OUTPUT%.mid}_chords.json"

echo "=== Output: $OUTPUT ==="
