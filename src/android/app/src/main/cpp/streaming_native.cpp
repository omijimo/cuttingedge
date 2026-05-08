#include <jni.h>

#include "pitch_autocorr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kNumRoots = 12;
constexpr int kNumQualities = 7;
constexpr int kNoChordId = kNumRoots * kNumQualities;  // 84
constexpr int kPadChordId = kNoChordId + 1;            // 85

struct PitchDetectorState {
    int sample_rate;
    float min_freq;
    float max_freq;
};


int clamp_midi(int v) {
    return std::max(0, std::min(127, v));
}

std::vector<int> chord_pitches(int chord_id, int octave) {
    if (chord_id < 0 || chord_id >= kNoChordId) return {};
    const int root = chord_id / kNumQualities;
    const int quality = chord_id % kNumQualities;
    const int base = octave * 12 + root;

    switch (quality) {
        case 0: return {base + 0, base + 4, base + 7};      // maj
        case 1: return {base + 0, base + 3, base + 7};      // min
        case 2: return {base + 0, base + 3, base + 6};      // dim
        case 3: return {base + 0, base + 4, base + 8};      // aug
        case 4: return {base + 0, base + 4, base + 7, base + 10};  // dom7
        case 5: return {base + 0, base + 4, base + 7, base + 11};  // maj7
        case 6: return {base + 0, base + 3, base + 7, base + 10};  // min7
        default: return {base + 0, base + 4, base + 7};
    }
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_accompaniment_streaming_NativePitchDetector_nativeCreate(
    JNIEnv*,
    jobject,
    jint sample_rate,
    jfloat min_freq,
    jfloat max_freq
) {
    auto* state = new PitchDetectorState{
        static_cast<int>(sample_rate),
        static_cast<float>(min_freq),
        static_cast<float>(max_freq),
    };
    return reinterpret_cast<jlong>(state);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_accompaniment_streaming_NativePitchDetector_nativeProcess(
    JNIEnv* env,
    jobject,
    jlong handle,
    jfloatArray frame
) {
    auto* state = reinterpret_cast<PitchDetectorState*>(handle);
    if (state == nullptr || frame == nullptr) {
        return nullptr;
    }

    const jsize n = env->GetArrayLength(frame);
    std::vector<float> x(static_cast<size_t>(n));
    env->GetFloatArrayRegion(frame, 0, n, x.data());

    auto [midi, confidence] = streaming_pitch::detect_midi_note(
        x,
        state->sample_rate,
        state->min_freq,
        state->max_freq
    );

    jfloat out_data[2] = {midi, confidence};
    jfloatArray out = env->NewFloatArray(2);
    env->SetFloatArrayRegion(out, 0, 2, out_data);
    return out;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_accompaniment_streaming_NativePitchDetector_nativeReset(
    JNIEnv*,
    jobject,
    jlong
) {
    // Stateless for now.
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_accompaniment_streaming_NativePitchDetector_nativeDestroy(
    JNIEnv*,
    jobject,
    jlong handle
) {
    auto* state = reinterpret_cast<PitchDetectorState*>(handle);
    delete state;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_accompaniment_streaming_NativeAccompanimentEngine_nativeGenerate(
    JNIEnv* env,
    jobject,
    jintArray chord_ids,
    jint start_beat,
    jint ticks_per_beat,
    jint octave,
    jint velocity
) {
    if (chord_ids == nullptr || ticks_per_beat <= 0) {
        return nullptr;
    }
    const jsize n = env->GetArrayLength(chord_ids);
    std::vector<jint> ids(static_cast<size_t>(n));
    env->GetIntArrayRegion(chord_ids, 0, n, ids.data());

    std::vector<jint> events;  // pitch, startTick, durationTick, velocity
    events.reserve(static_cast<size_t>(n) * 8);
    const jint duration = static_cast<jint>(std::max(1.0, std::floor(ticks_per_beat * 0.9)));

    for (jsize i = 0; i < n; ++i) {
        const int cid = ids[static_cast<size_t>(i)];
        if (cid == kNoChordId || cid == kPadChordId) {
            continue;
        }
        auto pitches = chord_pitches(cid, octave);
        if (pitches.empty()) continue;
        std::sort(pitches.begin(), pitches.end());
        const int bass = clamp_midi(pitches.front());
        const jint beat_index = start_beat + static_cast<jint>(i);
        const jint start_tick = beat_index * ticks_per_beat;

        if ((beat_index % 2) == 0) {
            events.push_back(bass);
            events.push_back(start_tick);
            events.push_back(duration);
            events.push_back(static_cast<jint>(std::max(0, std::min(127, velocity))));
        } else {
            for (size_t pi = 1; pi < pitches.size(); ++pi) {
                events.push_back(clamp_midi(pitches[pi]));
                events.push_back(start_tick);
                events.push_back(duration);
                events.push_back(static_cast<jint>(std::max(0, std::min(127, velocity - 10))));
            }
        }
    }

    jintArray out = env->NewIntArray(static_cast<jsize>(events.size()));
    env->SetIntArrayRegion(out, 0, static_cast<jsize>(events.size()), events.data());
    return out;
}
