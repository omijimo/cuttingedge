package com.example.accompaniment.streaming

enum class FrameState {
    ON,
    HOLD,
    REST,
}

data class MelodyFrame(
    val stepIndex: Long,
    val timestampNanos: Long,
    val midiPitch: Int,
    val state: FrameState,
    val confidence: Float,
) {
    val melodyToken: Int
        get() = when (state) {
            FrameState.ON -> midiPitch.mod(12)
            FrameState.HOLD -> 12
            FrameState.REST -> 13
        }
}

data class StreamingConfig(
    val bpm: Float,
    val beatsPerBar: Int = 4,
    val gridResolution: Int = 16,
    val sampleRateHz: Int = 16_000,
    val chordWindowSeconds: Float = 5.0f,
    val chordStrideSeconds: Float = 3.0f,
    val chordLagBeats: Int = 1,
    val accompanimentWindowChords: Int = 8,
    val scheduleBarLag: Int = 1,
)

data class ScheduledNote(
    val pitch: Int,
    val velocity: Int,
    val startTick: Long,
    val durationTicks: Long,
)
