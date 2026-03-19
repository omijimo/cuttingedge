"""Chord vocabulary: 12 roots x 7 qualities + no-chord + pad = 86 tokens."""

from __future__ import annotations

ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NUM_ROOTS = 12

QUALITIES = ["maj", "min", "dim", "aug", "dom7", "maj7", "min7"]
NUM_QUALITIES = 7

NO_CHORD_ID = NUM_ROOTS * NUM_QUALITIES   # 84
PAD_CHORD_ID = NO_CHORD_ID + 1            # 85
CHORD_VOCAB_SIZE = PAD_CHORD_ID + 1       # 86

# Enharmonic / alias normalization tables

ENHARMONIC_MAP: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#",
    "Bb": "A#", "Cb": "B", "B#": "C", "E#": "F",
    "D♭": "C#", "E♭": "D#", "F♭": "E", "G♭": "F#", "A♭": "G#",
    "B♭": "A#", "C♭": "B",
}

QUALITY_ALIASES: dict[str, str] = {
    "major": "maj", "M": "maj", "": "maj",
    "minor": "min", "m": "min", "-": "min",
    "diminished": "dim", "o": "dim", "°": "dim",
    "augmented": "aug", "+": "aug",
    "dominant7": "dom7", "7": "dom7",
    "major7": "maj7", "M7": "maj7", "Δ7": "maj7", "Maj7": "maj7",
    "minor7": "min7", "m7": "min7", "-7": "min7",
    "N": "N.C.", "N.C.": "N.C.", "NC": "N.C.", "none": "N.C.",
}

# Normalization helpers


def normalize_root(root: str) -> str:
    return ENHARMONIC_MAP.get(root, root)


def normalize_quality(quality: str) -> str:
    return QUALITY_ALIASES.get(quality, quality)


# Encode / decode


def encode_chord(root: str, quality: str) -> int:
    """Encode (root, quality) pair to a single chord id."""
    if quality in ("N.C.", "none", "N", "NC") or root in ("N.C.", "NC"):
        return NO_CHORD_ID
    root = normalize_root(root)
    quality = normalize_quality(quality)
    if root not in ROOTS or quality not in QUALITIES:
        return NO_CHORD_ID
    return ROOTS.index(root) * NUM_QUALITIES + QUALITIES.index(quality)


def decode_chord(chord_id: int) -> tuple[str, str]:
    """Decode a chord id to (root, quality). Returns ("N.C.","") or ("PAD","")."""
    if chord_id == NO_CHORD_ID:
        return ("N.C.", "")
    if chord_id == PAD_CHORD_ID:
        return ("PAD", "")
    if not (0 <= chord_id < NO_CHORD_ID):
        return ("N.C.", "")
    return (ROOTS[chord_id // NUM_QUALITIES], QUALITIES[chord_id % NUM_QUALITIES])


def chord_to_string(chord_id: int) -> str:
    root, quality = decode_chord(chord_id)
    if root in ("N.C.", "PAD"):
        return root
    return f"{root}{quality}"


# Pitch helpers

_INTERVALS: dict[str, list[int]] = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}


def chord_to_pitches(root: str, quality: str, octave: int = 4) -> list[int]:
    """Return MIDI pitches for a chord in the given octave."""
    if root in ("N.C.", "PAD", ""):
        return []
    root = normalize_root(root)
    if root not in ROOTS:
        return []
    base = ROOTS.index(root) + octave * 12
    offsets = _INTERVALS.get(quality, [0, 4, 7])
    return [base + o for o in offsets]


# Parse a chord label string


def parse_chord_label(label: str) -> tuple[str, str]:
    """Parse 'C#min7' or 'Gmaj' into (root, quality)."""
    label = label.strip()
    if not label or label in ("N.C.", "NC", "N", "none", "X"):
        return ("N.C.", "")

    root = label[0]
    rest = label[1:]
    if rest and rest[0] in "#b♭♯":
        root += rest[0]
        rest = rest[1:]

    root = normalize_root(root)
    quality = normalize_quality(rest)
    if quality == "N.C.":
        return ("N.C.", "")
    if quality not in QUALITIES:
        quality = "maj"
    return (root, quality)
