"""Evaluation and checkpoint loading."""

from __future__ import annotations

import logging as _logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..models.chord_transformer import ChordTransformer
from ..data.dataset import MelodyChordDataset
from ..data.collate import chord_collate_fn
from ..training.losses import ChordPredictionLoss
from ..training.metrics import compute_metrics

logger = _logging.getLogger("accompaniment")


def load_model_from_checkpoint(
    ckpt_path: str | Path,
    device: torch.device | None = None,
) -> tuple[ChordTransformer, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]

    model = ChordTransformer(
        melody_vocab_size=m["melody_vocab_size"],
        chord_vocab_size=m["chord_vocab_size"],
        d_model=m["d_model"],
        nhead=m["nhead"],
        num_layers=m["num_layers"],
        dim_feedforward=m["dim_feedforward"],
        dropout=m.get("dropout", 0.1),
        max_seq_len=m["max_seq_len"],
        num_bar_positions=m.get("num_bar_positions", 16),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def evaluate(cfg: dict, ckpt_path: str | Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model_from_checkpoint(ckpt_path, device)

    val_ds = MelodyChordDataset(cfg["data"]["data_dir"], cfg["model"]["max_seq_len"], "val")
    val_dl = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, collate_fn=chord_collate_fn, num_workers=0,
    )

    criterion = ChordPredictionLoss(pad_id=cfg["model"]["chord_vocab_size"] - 1)
    total_loss = 0.0
    steps = 0
    all_preds: list[torch.Tensor] = []
    all_tgts: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_dl:
            mel = batch["melody_tokens"].to(device)
            bpos = batch["beat_positions"].to(device)
            tgt = batch["chord_targets"].to(device)
            mask = batch["padding_mask"].to(device)

            logits = model(mel, bpos, mask)
            total_loss += criterion(logits, tgt).item()
            steps += 1
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_tgts.append(tgt.cpu())

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_tgts))
    metrics["val_loss"] = total_loss / max(steps, 1)

    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)
    return metrics
