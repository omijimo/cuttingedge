#include "pitch_autocorr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace streaming_pitch {
namespace {

float compute_rms(const std::vector<float>& x) {
    if (x.empty()) return 0.0f;
    double s = 0.0;
    for (float v : x) {
        s += static_cast<double>(v) * static_cast<double>(v);
    }
    return static_cast<float>(std::sqrt(s / static_cast<double>(x.size())));
}

float parabolic_refine(const std::vector<float>& v, int i) {
    if (i <= 0 || i + 1 >= static_cast<int>(v.size())) return static_cast<float>(i);
    const float y0 = v[i - 1];
    const float y1 = v[i];
    const float y2 = v[i + 1];
    const float denom = (y0 - 2.0f * y1 + y2);
    if (std::fabs(denom) < 1e-9f) return static_cast<float>(i);
    const float delta = 0.5f * (y0 - y2) / denom;
    return static_cast<float>(i) + std::clamp(delta, -0.5f, 0.5f);
}

}  // namespace

std::pair<float, float> detect_midi_note(
    const std::vector<float>& frame,
    int sample_rate,
    float min_freq_hz,
    float max_freq_hz,
    float silence_rms_threshold
) {
    if (frame.size() < 128 || sample_rate <= 0 || min_freq_hz <= 0.0f || max_freq_hz <= min_freq_hz) {
        return {-1.0f, 0.0f};
    }

    const float rms = compute_rms(frame);
    if (rms < silence_rms_threshold) {
        return {-1.0f, 0.0f};
    }

    std::vector<float> x(frame.begin(), frame.end());
    float mean = 0.0f;
    for (float v : x) mean += v;
    mean /= static_cast<float>(x.size());
    for (float& v : x) v -= mean;

    const int min_lag = std::max(1, static_cast<int>(std::floor(static_cast<float>(sample_rate) / max_freq_hz)));
    const int max_lag = std::min(
        static_cast<int>(x.size()) - 2,
        static_cast<int>(std::ceil(static_cast<float>(sample_rate) / min_freq_hz))
    );
    if (max_lag <= min_lag) {
        return {-1.0f, 0.0f};
    }

    std::vector<float> norm_corr(static_cast<size_t>(max_lag + 1), 0.0f);
    float best_score = -std::numeric_limits<float>::infinity();
    int best_lag = -1;

    for (int lag = min_lag; lag <= max_lag; ++lag) {
        double num = 0.0;
        double e0 = 0.0;
        double e1 = 0.0;
        for (size_t i = 0; i + static_cast<size_t>(lag) < x.size(); ++i) {
            const double a = static_cast<double>(x[i]);
            const double b = static_cast<double>(x[i + lag]);
            num += a * b;
            e0 += a * a;
            e1 += b * b;
        }
        if (e0 <= 1e-12 || e1 <= 1e-12) continue;
        const float score = static_cast<float>(num / std::sqrt(e0 * e1));
        norm_corr[static_cast<size_t>(lag)] = score;
        if (score > best_score) {
            best_score = score;
            best_lag = lag;
        }
    }

    if (best_lag < 0 || best_score < 0.30f) {
        return {-1.0f, std::max(0.0f, best_score)};
    }

    // For monophonic singing/instrument input, prefer the first strong local peak.
    // This suppresses octave errors where a higher-lag harmonic wins globally.
    constexpr float kStrongPeak = 0.55f;
    int selected_lag = best_lag;
    for (int lag = min_lag + 1; lag < max_lag; ++lag) {
        const float c_prev = norm_corr[static_cast<size_t>(lag - 1)];
        const float c_cur = norm_corr[static_cast<size_t>(lag)];
        const float c_next = norm_corr[static_cast<size_t>(lag + 1)];
        if (c_cur >= kStrongPeak && c_cur >= c_prev && c_cur >= c_next) {
            selected_lag = lag;
            break;
        }
    }

    const float refined_lag = parabolic_refine(norm_corr, selected_lag);
    if (refined_lag <= 0.0f) {
        return {-1.0f, best_score};
    }
    const float freq_hz = static_cast<float>(sample_rate) / refined_lag;
    if (!(freq_hz >= min_freq_hz && freq_hz <= max_freq_hz)) {
        return {-1.0f, best_score};
    }

    const float midi = 69.0f + 12.0f * std::log2(freq_hz / 440.0f);
    const float confidence = std::clamp(best_score, 0.0f, 1.0f);
    return {midi, confidence};
}

}  // namespace streaming_pitch
