"""End-to-end test: preprocess → train → infer → export on synthetic data."""

from __future__ import annotations

from pathlib import Path

import torch
import numpy as np

from accompaniment.data.preprocess import generate_synthetic_dataset
from accompaniment.data.dataset import MelodyChordDataset
from accompaniment.models.chord_transformer import ChordTransformer
from accompaniment.training.losses import ChordPredictionLoss
from accompaniment.training.metrics import compute_metrics
from accompaniment.generation.chord_decode import greedy_decode, decode_to_labels
from accompaniment.generation.accompaniment_rules import generate_accompaniment
from accompaniment.generation.midi_render import render_output
from accompaniment.io.tokenization import MelodyEvent


def test_synthetic_data_generation(tmp_path: Path):
    out = tmp_path / "synth"
    generate_synthetic_dataset(out, num_examples=10, seed=0)
    assert (out / "train.json").exists()
    assert (out / "val.json").exists()
    assert len(list((out / "train").glob("*.npz"))) == 9
    assert len(list((out / "val").glob("*.npz"))) == 1


def test_dataset_loading(tmp_path: Path):
    out = tmp_path / "synth"
    generate_synthetic_dataset(out, num_examples=10, seed=0)
    ds = MelodyChordDataset(out, max_seq_len=64, split="train")
    assert len(ds) == 9
    item = ds[0]
    assert item["melody_tokens"].shape == (64,)
    assert item["chord_targets"].shape == (64,)
    assert item["padding_mask"].dtype == torch.bool


def test_train_one_step(tiny_config: dict, tmp_path: Path):
    cfg = tiny_config
    cfg["data"]["data_dir"] = str(tmp_path / "data")
    generate_synthetic_dataset(cfg["data"]["data_dir"], num_examples=20, seed=42)

    ds = MelodyChordDataset(cfg["data"]["data_dir"], cfg["model"]["max_seq_len"], "train")
    batch = {k: v.unsqueeze(0) for k, v in ds[0].items()}

    model = ChordTransformer(
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=0.0,
        max_seq_len=cfg["model"]["max_seq_len"],
    )
    model.train()

    logits = model(batch["melody_tokens"], batch["beat_positions"], batch["padding_mask"])
    loss_fn = ChordPredictionLoss()
    loss = loss_fn(logits, batch["chord_targets"])
    loss.backward()
    assert loss.item() > 0


def test_inference_pipeline(tiny_config: dict, tmp_path: Path):
    """Tiny end-to-end: model forward → decode → accompaniment → MIDI."""
    cfg = tiny_config
    model = ChordTransformer(
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=0.0,
        max_seq_len=cfg["model"]["max_seq_len"],
    )
    model.eval()

    seq_len = 32
    mel = torch.randint(0, 14, (1, seq_len))
    beats = torch.randint(0, 16, (1, seq_len))
    with torch.no_grad():
        logits = model(mel, beats)

    chord_ids = greedy_decode(logits, grid_resolution=16, steps_per_beat=4)
    labels = decode_to_labels(chord_ids)
    assert len(labels) > 0

    accomp = generate_accompaniment(chord_ids, tempo=120.0, pattern_name="block")
    assert isinstance(accomp, list)

    # Build dummy melody events for rendering
    events = [MelodyEvent(60, "on", 0, i % 16) for i in range(seq_len)]
    pm = render_output(events, accomp, tempo=120.0, grid_resolution=16)
    assert len(pm.instruments) == 2


def test_onnx_export(tiny_config: dict, tmp_path: Path):
    """Export to ONNX and verify output shapes."""
    cfg = tiny_config
    cfg["data"]["data_dir"] = str(tmp_path / "data")
    generate_synthetic_dataset(cfg["data"]["data_dir"], num_examples=10, seed=42)

    model = ChordTransformer(
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_layers=cfg["model"]["num_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=0.0,
        max_seq_len=cfg["model"]["max_seq_len"],
    )

    ckpt_path = tmp_path / "test.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

    from accompaniment.export.export_onnx import export_onnx

    onnx_path = tmp_path / "model.onnx"
    export_onnx(ckpt_path, onnx_path, cfg)
    assert onnx_path.exists()

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        max_seq = cfg["model"]["max_seq_len"]
        out = session.run(
            None,
            {
                "melody_tokens": np.zeros((1, max_seq), dtype=np.int64),
                "beat_positions": np.zeros((1, max_seq), dtype=np.int64),
            },
        )
        assert out[0].shape == (1, max_seq, 86)
    except ImportError:
        pass  # onnxruntime optional for CI
