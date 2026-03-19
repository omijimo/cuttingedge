#!/usr/bin/env bash
set -euo pipefail

echo "=== Preprocessing synthetic data ==="
python -m accompaniment.cli preprocess --config configs/small.yaml

echo "=== Training (small config) ==="
python -m accompaniment.cli train --config configs/small.yaml

echo "=== Done ==="
echo "Checkpoint: outputs/checkpoints/best.pt"
