#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/checkpoints/best.pt}"
OUTDIR="${2:-exports}"

echo "=== Exporting models ==="
python -m accompaniment.cli export \
    --checkpoint "$CHECKPOINT" \
    --outdir "$OUTDIR"

echo "=== Verifying ONNX ==="
python -c "
from accompaniment.export.verify_exports import verify_onnx
from accompaniment.utils.logging import setup_logging
setup_logging()
ok = verify_onnx('$CHECKPOINT', '$OUTDIR/chord_model.onnx')
raise SystemExit(0 if ok else 1)
"

echo "=== Verifying TFLite ==="
python -c "
from accompaniment.export.verify_exports import verify_tflite
from accompaniment.utils.logging import setup_logging
setup_logging()
ok = verify_tflite('$CHECKPOINT', '$OUTDIR/chord_model.tflite')
raise SystemExit(0 if ok else 1)
" || echo "(TFLite verification skipped or failed)"

echo "=== Done. Exports in $OUTDIR ==="
