package com.example.accompaniment.streaming

import kotlin.math.roundToInt

/**
 * JNI wrapper around native autocorrelation pitch detection.
 *
 * The native layer returns [midiFloat, confidence] for each PCM frame.
 * This class then maps detector output into quantized MelodyFrame events.
 */
class NativePitchDetector(
    private val config: StreamingConfig,
    minFreqHz: Float = 65.4f,
    maxFreqHz: Float = 1046.5f,
    private val onThreshold: Float = 0.35f,
    private val holdThreshold: Float = 0.20f,
) : AutoCloseable {

    private val nanosPerBeat = (60_000_000_000.0 / config.bpm).toLong()
    private val stepsPerBeat = (config.gridResolution / config.beatsPerBar).coerceAtLeast(1)
    private val nanosPerStep = (nanosPerBeat / stepsPerBeat.toDouble()).toLong()

    private var handle: Long = nativeCreate(config.sampleRateHz, minFreqHz, maxFreqHz)
    private var lastPitch: Int = -1

    fun processFrame(pcm: FloatArray, timestampNanos: Long): MelodyFrame {
        val out = nativeProcess(handle, pcm)
        val midiFloat = out[0]
        val confidence = out[1]
        val step = (timestampNanos / nanosPerStep).coerceAtLeast(0)

        val midi = if (midiFloat >= 0f) midiFloat.roundToInt().coerceIn(0, 127) else -1
        val state = when {
            confidence >= onThreshold && midi >= 0 -> {
                if (lastPitch >= 0 && kotlin.math.abs(midi - lastPitch) <= 1) FrameState.HOLD else FrameState.ON
            }
            confidence >= holdThreshold && lastPitch >= 0 -> FrameState.HOLD
            else -> FrameState.REST
        }

        val frame = MelodyFrame(
            stepIndex = step,
            timestampNanos = timestampNanos,
            midiPitch = if (state == FrameState.REST) -1 else (if (midi >= 0) midi else lastPitch),
            state = state,
            confidence = confidence,
        )

        lastPitch = if (frame.state == FrameState.REST) -1 else frame.midiPitch
        return frame
    }

    fun reset() {
        nativeReset(handle)
        lastPitch = -1
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private external fun nativeCreate(sampleRate: Int, minFreqHz: Float, maxFreqHz: Float): Long
    private external fun nativeProcess(handle: Long, frame: FloatArray): FloatArray
    private external fun nativeReset(handle: Long)
    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("accompaniment_streaming_native")
        }
    }
}
