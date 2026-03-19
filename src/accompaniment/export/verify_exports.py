"""Smoke-test exported models against PyTorch reference outputs."""

from __future__ import annotations

import logging as _logging
from pathlib import Path

import numpy as np
import torch

from ..training.eval import load_model_from_checkpoint

logger = _logging.getLogger("accompaniment")


def _random_inputs(max_seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    mel = torch.randint(0, 14, (1, max_seq))
    beats = torch.randint(0, 16, (1, max_seq))
    return mel, beats


def verify_onnx(
    ckpt_path: str | Path,
    onnx_path: str | Path,
    atol: float = 1e-4,
) -> bool:
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError:
        logger.error("onnxruntime not installed — cannot verify ONNX export")
        return False

    model, mcfg = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
    mel, beats = _random_inputs(mcfg["model"]["max_seq_len"])

    with torch.no_grad():
        pt_out = model(mel, beats).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(
        None,
        {"melody_tokens": mel.numpy(), "beat_positions": beats.numpy()},
    )[0]

    ok = bool(np.allclose(pt_out, onnx_out, atol=atol))
    diff = float(np.max(np.abs(pt_out - onnx_out)))
    logger.info("ONNX verify: %s  (max_diff=%.6f)", "PASS" if ok else "FAIL", diff)
    return ok


def verify_coreml(
    ckpt_path: str | Path,
    coreml_path: str | Path,
    atol: float = 1e-3,
) -> bool:
    try:
        import coremltools as ct  # type: ignore[import-untyped]
    except ImportError:
        logger.error("coremltools not installed — cannot verify Core ML export")
        return False

    model, mcfg = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
    mel, beats = _random_inputs(mcfg["model"]["max_seq_len"])

    with torch.no_grad():
        pt_out = model(mel, beats).numpy()

    ml_model = ct.models.MLModel(str(coreml_path))
    pred = ml_model.predict({
        "melody_tokens": mel.numpy().astype(np.int32),
        "beat_positions": beats.numpy().astype(np.int32),
    })
    cm_out = list(pred.values())[0]

    ok = bool(np.allclose(pt_out, cm_out, atol=atol))
    diff = float(np.max(np.abs(pt_out - cm_out)))
    logger.info("Core ML verify: %s  (max_diff=%.6f)", "PASS" if ok else "FAIL", diff)
    return ok


def verify_tflite(
    ckpt_path: str | Path,
    tflite_path: str | Path,
    atol: float = 1e-4,
) -> bool:
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-untyped]
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "Neither ai-edge-litert nor tflite-runtime installed "
                "— cannot verify TFLite export"
            )
            return False

    model, mcfg = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
    mel, beats = _random_inputs(mcfg["model"]["max_seq_len"])

    with torch.no_grad():
        pt_out = model(mel, beats).numpy()

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()

    interp.set_tensor(inp[0]["index"], mel.numpy())
    interp.set_tensor(inp[1]["index"], beats.numpy())
    interp.invoke()
    tfl_out = interp.get_tensor(out[0]["index"])

    ok = bool(np.allclose(pt_out, tfl_out, atol=atol))
    diff = float(np.max(np.abs(pt_out - tfl_out)))
    logger.info("TFLite verify: %s  (max_diff=%.6f)", "PASS" if ok else "FAIL", diff)
    return ok
