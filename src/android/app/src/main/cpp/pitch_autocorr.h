#pragma once

#include <utility>
#include <vector>

namespace streaming_pitch {

// Returns {midi_note_float, confidence}. midi_note_float < 0 means no pitch.
std::pair<float, float> detect_midi_note(
    const std::vector<float>& frame,
    int sample_rate,
    float min_freq_hz,
    float max_freq_hz,
    float silence_rms_threshold = 0.01f
);

}  // namespace streaming_pitch
