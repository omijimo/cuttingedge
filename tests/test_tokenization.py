"""Tests for tokenization and chord vocabulary."""

import numpy as np

from accompaniment.io.chord_vocab import (
    encode_chord,
    decode_chord,
    chord_to_string,
    chord_to_pitches,
    parse_chord_label,
    normalize_root,
    NO_CHORD_ID,
    PAD_CHORD_ID,
    CHORD_VOCAB_SIZE,
)
from accompaniment.io.tokenization import (
    MelodyEvent,
    melody_event_to_token,
    tokenize_melody,
    pad_sequence,
    HOLD_TOKEN,
    REST_TOKEN,
    PAD_TOKEN,
    MELODY_VOCAB_SIZE,
)


# ---- chord vocab --------------------------------------------------------


def test_encode_decode_roundtrip():
    for root in ["C", "D", "F#", "A#"]:
        for q in ["maj", "min", "dom7"]:
            cid = encode_chord(root, q)
            assert 0 <= cid < NO_CHORD_ID
            r, qd = decode_chord(cid)
            assert r == root and qd == q


def test_no_chord():
    assert encode_chord("N.C.", "") == NO_CHORD_ID
    assert decode_chord(NO_CHORD_ID) == ("N.C.", "")


def test_enharmonic():
    assert normalize_root("Db") == "C#"
    assert normalize_root("Bb") == "A#"
    assert encode_chord("Db", "min") == encode_chord("C#", "min")


def test_chord_to_pitches():
    pitches = chord_to_pitches("C", "maj", 4)
    assert pitches == [48, 52, 55]


def test_parse_chord_label():
    assert parse_chord_label("Cmaj") == ("C", "maj")
    assert parse_chord_label("F#min7") == ("F#", "min7")
    assert parse_chord_label("Bb7")[1] == "dom7"
    assert parse_chord_label("N.C.") == ("N.C.", "")


def test_vocab_size():
    assert CHORD_VOCAB_SIZE == 86
    assert MELODY_VOCAB_SIZE == 15


# ---- melody tokenization -----------------------------------------------


def test_melody_event_to_token():
    assert melody_event_to_token(MelodyEvent(60, "on", 0, 0)) == 0   # C
    assert melody_event_to_token(MelodyEvent(61, "on", 0, 0)) == 1   # C#
    assert melody_event_to_token(MelodyEvent(-1, "rest", 0, 0)) == REST_TOKEN
    assert melody_event_to_token(MelodyEvent(60, "hold", 0, 0)) == HOLD_TOKEN


def test_tokenize_melody():
    events = [
        MelodyEvent(60, "on", 0, 0),
        MelodyEvent(60, "hold", 0, 1),
        MelodyEvent(-1, "rest", 0, 2),
        MelodyEvent(64, "on", 0, 3),
    ]
    tokens, positions = tokenize_melody(events)
    assert tokens.tolist() == [0, HOLD_TOKEN, REST_TOKEN, 4]
    assert positions.tolist() == [0, 1, 2, 3]


def test_pad_sequence():
    arr = np.array([1, 2, 3], dtype=np.int64)
    padded = pad_sequence(arr, 6, PAD_TOKEN)
    assert padded.tolist() == [1, 2, 3, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]

    truncated = pad_sequence(arr, 2, PAD_TOKEN)
    assert truncated.tolist() == [1, 2]
