"""Training loop for the melody → chord transformer."""

from __future__ import annotations

import json
import time
import logging as _logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.chord_transformer import ChordTransformer
from ..data.dataset import MelodyChordDataset
from ..data.collate import chord_collate_fn
from ..training.losses import ChordPredictionLoss
from ..training.metrics import compute_metrics
from ..utils.seed import set_seed

logger = _logging.getLogger("accompaniment")


def train(cfg: dict) -> Path:
    """Run training and return path to best checkpoint."""
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    # ---- data -----------------------------------------------------------
    train_ds = MelodyChordDataset(dcfg["data_dir"], mcfg["max_seq_len"], "train")
    val_ds = MelodyChordDataset(dcfg["data_dir"], mcfg["max_seq_len"], "val")
    logger.info("Train: %d  Val: %d", len(train_ds), len(val_ds))

    train_dl = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        collate_fn=chord_collate_fn, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        collate_fn=chord_collate_fn, num_workers=0,
    )

    # ---- model ----------------------------------------------------------
    model = ChordTransformer(
        melody_vocab_size=mcfg["melody_vocab_size"],
        chord_vocab_size=mcfg["chord_vocab_size"],
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        num_layers=mcfg["num_layers"],
        dim_feedforward=mcfg["dim_feedforward"],
        dropout=mcfg["dropout"],
        max_seq_len=mcfg["max_seq_len"],
        num_bar_positions=mcfg.get("num_bar_positions", 16),
    ).to(device)
    logger.info("Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = ChordPredictionLoss(pad_id=mcfg["chord_vocab_size"] - 1)
    optimizer = AdamW(
        model.parameters(), lr=tcfg["learning_rate"],
        weight_decay=tcfg.get("weight_decay", 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tcfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_ctr = 0
    history: list[dict] = []

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()

        # ---- train epoch ------------------------------------------------
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in train_dl:
            mel = batch["melody_tokens"].to(device)
            bpos = batch["beat_positions"].to(device)
            tgt = batch["chord_targets"].to(device)
            mask = batch["padding_mask"].to(device)

            logits = model(mel, bpos, mask)
            loss = criterion(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train = train_loss / max(train_steps, 1)

        # ---- val epoch --------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds: list[torch.Tensor] = []
        all_tgts: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_dl:
                mel = batch["melody_tokens"].to(device)
                bpos = batch["beat_positions"].to(device)
                tgt = batch["chord_targets"].to(device)
                mask = batch["padding_mask"].to(device)

                logits = model(mel, bpos, mask)
                loss = criterion(logits, tgt)
                val_loss += loss.item()
                val_steps += 1

                all_preds.append(logits.argmax(dim=-1).cpu())
                all_tgts.append(tgt.cpu())

        avg_val = val_loss / max(val_steps, 1)
        metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_tgts))

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
            "chord_acc=%.3f  root_acc=%.3f  qual_acc=%.3f  (%.1fs)",
            epoch, tcfg["epochs"], avg_train, avg_val,
            metrics["chord_accuracy"], metrics["root_accuracy"],
            metrics["quality_accuracy"], elapsed,
        )
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val, **metrics})

        # ---- checkpoint -------------------------------------------------
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val,
            "metrics": metrics,
            "config": cfg,
        }
        torch.save(ckpt_payload, ckpt_dir / "last.pt")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_ctr = 0
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            logger.info("  → saved best (val_loss=%.4f)", avg_val)
        else:
            patience_ctr += 1

        if patience_ctr >= tcfg["patience"]:
            logger.info("Early stopping at epoch %d", epoch)
            break

    with open(log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    return ckpt_dir / "best.pt"
