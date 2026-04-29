import torch
import numpy as np
from pathlib import Path

from accompaniment.models.chord_transformer import ChordTransformer


# ====== 1. load model ======
ckpt = torch.load("outputs/checkpoints/best.pt", map_location="cpu")

model = ChordTransformer(
    melody_vocab_size=15,
    chord_vocab_size=86,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    max_seq_len=256,
    num_bar_positions=16,
)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded.")


# ====== 2. load one example ======
npz_path = "data/processed/val/val_00000.npz"
data = np.load(npz_path)

max_len = 256

melody = torch.tensor(data["melody_tokens"][:max_len]).unsqueeze(0)
beat = torch.tensor(data["beat_positions"][:max_len]).unsqueeze(0)
target = data["chord_targets"][:max_len]

print("Example loaded.")


# ====== 3. run model ======
with torch.no_grad():
    logits = model(melody, beat)
    pred = logits.argmax(dim=-1).squeeze(0).numpy()


# ====== 4. compare ======
print("\nFirst 50 predictions vs ground truth:\n")

for i in range(50):
    print(f"{i:02d}: pred={pred[i]:3d} | true={target[i]:3d}")