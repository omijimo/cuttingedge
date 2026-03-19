"""Export chord transformer to ONNX."""

from __future__ import annotations

from pathlib import Path
import torch

from ..training.eval import load_model_from_checkpoint


def export_onnx(
    ckpt_path: str | Path,
    output_path: str | Path,
    cfg: dict | None = None,
    opset_version: int = 14,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, mcfg = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
    max_seq = mcfg["model"]["max_seq_len"]

    dummy_mel = torch.zeros(1, max_seq, dtype=torch.long)
    dummy_beats = torch.zeros(1, max_seq, dtype=torch.long)

    if cfg and "export" in cfg:
        opset_version = cfg["export"].get("opset_version", opset_version)

    batch = torch.export.Dim("batch", min=1, max=32)
    seq = torch.export.Dim("seq_len", min=2, max=max_seq)
    dynamic_shapes = {
        "melody_tokens": {0: batch, 1: seq},
        "beat_positions": {0: batch, 1: seq},
    }

    torch.onnx.export(
        model,
        (dummy_mel, dummy_beats),
        str(output_path),
        opset_version=opset_version,
        input_names=["melody_tokens", "beat_positions"],
        output_names=["chord_logits"],
        dynamic_shapes=dynamic_shapes,
    )
    return output_path
