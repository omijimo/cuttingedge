"""Tests for model forward pass shapes and basic behavior."""

import torch

from accompaniment.models.chord_transformer import ChordTransformer


def test_forward_shape():
    model = ChordTransformer(
        melody_vocab_size=15,
        chord_vocab_size=86,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=64,
    )
    mel = torch.randint(0, 15, (2, 32))
    beats = torch.randint(0, 16, (2, 32))
    logits = model(mel, beats)
    assert logits.shape == (2, 32, 86)


def test_forward_with_padding_mask():
    model = ChordTransformer(
        melody_vocab_size=15,
        chord_vocab_size=86,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=64,
    )
    mel = torch.randint(0, 15, (2, 32))
    beats = torch.randint(0, 16, (2, 32))
    mask = torch.zeros(2, 32, dtype=torch.bool)
    mask[0, 16:] = True
    mask[1, 24:] = True

    logits = model(mel, beats, mask)
    assert logits.shape == (2, 32, 86)


def test_model_param_count():
    model = ChordTransformer(d_model=64, nhead=2, num_layers=1, dim_feedforward=128)
    n = sum(p.numel() for p in model.parameters())
    # Should be well under 1M for edge deployment
    assert n < 500_000
