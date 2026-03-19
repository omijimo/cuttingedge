"""Export chord transformer to Core ML (.mlpackage)."""

from __future__ import annotations

from pathlib import Path
import torch

from ..training.eval import load_model_from_checkpoint


def export_coreml(
    ckpt_path: str | Path,
    output_path: str | Path,
    cfg: dict | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import coremltools as ct  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "coremltools is not installed. Install with: pip install coremltools"
        )

    model, mcfg = load_model_from_checkpoint(ckpt_path, torch.device("cpu"))
    max_seq = mcfg["model"]["max_seq_len"]

    dummy_mel = torch.zeros(1, max_seq, dtype=torch.long)
    dummy_beats = torch.zeros(1, max_seq, dtype=torch.long)

    traced = torch.jit.trace(model, (dummy_mel, dummy_beats))

    ml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="melody_tokens", shape=(1, max_seq), dtype=int),
            ct.TensorType(name="beat_positions", shape=(1, max_seq), dtype=int),
        ],
        convert_to="mlprogram",
    )
    ml_model.save(str(output_path))
    return output_path
