"""Export chord transformer to TFLite via Google's litert-torch.

Primary path (recommended):
    PyTorch model  →  litert_torch.convert  →  .tflite

Fallback path (legacy, requires onnx-tf + tensorflow):
    PyTorch  →  ONNX  →  TF SavedModel  →  TFLite

litert-torch (formerly ai-edge-torch) uses torch.export under the hood,
so the model must be export-friendly (no data-dependent control flow,
no unsupported custom ops).
"""

from __future__ import annotations

import logging as _logging
from pathlib import Path

import torch

from ..training.eval import load_model_from_checkpoint

logger = _logging.getLogger("accompaniment")


def _export_via_litert(
    model: torch.nn.Module,
    sample_inputs: tuple[torch.Tensor, ...],
    output_path: Path,
) -> Path:
    """Primary path: direct PyTorch → TFLite via litert-torch."""
    try:
        import litert_torch  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "litert-torch is not installed. Install with:\n"
            "  pip install litert-torch\n"
            "See https://github.com/google-ai-edge/litert-torch"
        )

    edge_model = litert_torch.convert(model, sample_inputs)
    edge_model.export(str(output_path))
    return output_path


def _export_via_onnx_tf(
    ckpt_path: str | Path,
    output_path: Path,
    cfg: dict | None,
) -> Path:
    """Legacy fallback: PyTorch → ONNX → TF → TFLite."""
    import tempfile
    from .export_onnx import export_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"
        export_onnx(ckpt_path, onnx_path, cfg)

        try:
            import onnx  # type: ignore[import-untyped]
            from onnx_tf.backend import prepare  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "onnx-tf is not installed. Install with:  pip install onnx-tf\n"
                "This is the legacy fallback; prefer litert-torch instead."
            )

        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_path = str(Path(tmpdir) / "tf_saved_model")
        tf_rep.export_graph(tf_path)

        try:
            import tensorflow as tf  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "tensorflow is not installed. Install with:  pip install tensorflow"
            )

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_bytes = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_bytes)

    return output_path


def export_tflite(
    ckpt_path: str | Path,
    output_path: str | Path,
    cfg: dict | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, mcfg = load_model_from_checkpoint(ckpt_path, device)
    max_seq = mcfg["model"]["max_seq_len"]

    dummy_mel = torch.zeros(1, max_seq, dtype=torch.long)
    dummy_beats = torch.zeros(1, max_seq, dtype=torch.long)

    # Primary path: litert-torch
    try:
        _export_via_litert(model, (dummy_mel, dummy_beats), output_path)
        logger.info("TFLite model (litert-torch) → %s", output_path)
        return output_path
    except ImportError:
        logger.warning("litert-torch not available, trying legacy onnx-tf path")
    except Exception as exc:
        logger.warning("litert-torch conversion failed: %s", exc)
        logger.warning("Falling back to legacy onnx-tf path")

    # Fallback: ONNX → TF → TFLite
    _export_via_onnx_tf(ckpt_path, output_path, cfg)
    logger.info("TFLite model (onnx-tf) → %s", output_path)
    return output_path
