# Android Demo — Accompaniment

A minimal Android app that runs the chord prediction TFLite model on-device,
then generates a MIDI accompaniment and plays it back.

## What it does

1. Loads a sample MIDI melody from built-in assets
2. Parses the MIDI and tokenizes the melody (same scheme as Python training)
3. Runs the TFLite chord model on-device (CPU, ~2ms on modern phones)
4. Generates a bass + chord accompaniment pattern from predicted chords
5. Writes a two-track MIDI (melody + accompaniment) and plays it

## Setup

### Prerequisites

- Android Studio Ladybug (2024.2) or newer
- Android SDK 35
- JDK 17

### Steps

1. **Train and export the model** (from the repo root):

```bash
pip install -e ".[dev,tflite]"
python -m accompaniment.cli preprocess --config configs/small.yaml
python -m accompaniment.cli train --config configs/small.yaml
python -m accompaniment.cli export --checkpoint outputs/checkpoints/best.pt
```

2. **Copy the TFLite model into assets** (already done if you cloned after export):

```bash
cp exports/chord_model.tflite src/android/app/src/main/assets/
```

3. **Open in Android Studio:**
   - File → Open → select `src/android/`
   - Wait for Gradle sync
   - Run on emulator or device (API 26+)

### Sample MIDIs

Four built-in melodies are included in `app/src/main/assets/samples/`:

| File | Description |
|------|-------------|
| `c_major_scale.mid` | C major scale up and down (2 bars, 120 BPM) |
| `twinkle_melody.mid` | Twinkle Twinkle-style melody (4 bars, 100 BPM) |
| `blues_phrase.mid` | Short blues phrase (2 bars, 90 BPM) |
| `arpeggio_melody.mid` | Fast arpeggio pattern (4 bars, 140 BPM) |

## Architecture

```
MainActivity
  ├── MidiParser          — reads .mid bytes into note events
  ├── MelodyTokenizer     — quantizes to 16th-note grid, produces token arrays
  ├── ChordInferenceEngine — loads TFLite model, runs forward pass
  ├── ChordVocab          — decodes chord IDs to root+quality
  ├── AccompanimentGenerator — deterministic bass+chord pattern
  ├── MidiWriter          — writes two-track .mid file
  ├── MidiPlayer          — Android MediaPlayer for full MIDI playback
  └── RealtimeMidiPlayer  — low-latency MIDI clip playback for live accompaniment
```

All inference runs on a background coroutine; the UI stays responsive.
