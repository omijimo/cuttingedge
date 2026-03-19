"""PyTorch dataset for melody → chord training pairs."""

from __future__ import annotations

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from ..io.tokenization import PAD_TOKEN
from ..io.chord_vocab import PAD_CHORD_ID


class MelodyChordDataset(Dataset):
    """Loads preprocessed .npz examples indexed by a JSON manifest."""

    def __init__(
        self,
        data_dir: str | Path,
        max_seq_len: int = 256,
        split: str = "train",
    ):
        self.max_seq_len = max_seq_len
        self.data_dir = Path(data_dir)
        self.examples: list[dict] = []

        index_file = self.data_dir / f"{split}.json"
        if index_file.exists():
            with open(index_file) as f:
                self.examples = json.load(f)
        else:
            npz_dir = self.data_dir / split
            if npz_dir.is_dir():
                for p in sorted(npz_dir.glob("*.npz")):
                    self.examples.append({"path": str(p)})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.examples[idx]["path"])

        mel = data["melody_tokens"].astype(np.int64)
        bpos = data["beat_positions"].astype(np.int64)
        chords = data["chord_targets"].astype(np.int64)

        seq_len = min(len(mel), self.max_seq_len)

        m = np.full(self.max_seq_len, PAD_TOKEN, dtype=np.int64)
        b = np.zeros(self.max_seq_len, dtype=np.int64)
        c = np.full(self.max_seq_len, PAD_CHORD_ID, dtype=np.int64)

        m[:seq_len] = mel[:seq_len]
        b[:seq_len] = bpos[:seq_len]
        c[:seq_len] = chords[:seq_len]

        pad_mask = np.ones(self.max_seq_len, dtype=bool)
        pad_mask[:seq_len] = False

        return {
            "melody_tokens": torch.from_numpy(m),
            "beat_positions": torch.from_numpy(b),
            "chord_targets": torch.from_numpy(c),
            "padding_mask": torch.from_numpy(pad_mask),
        }
