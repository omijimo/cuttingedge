package com.example.accompaniment.streaming

import com.example.accompaniment.inference.ChordInferenceEngine

/**
 * Streaming chord engine with:
 * - circular melody-frame buffer
 * - overlapping sliding windows
 * - lagged/stabilized chord timeline suitable for low-latency accompaniment
 */
class StreamingChordEngine(
    private val config: StreamingConfig,
    private val inferenceEngine: ChordInferenceEngine,
    private val onInferenceDebug: ((InferenceDebugSnapshot) -> Unit)? = null,
) {

    private val stepsPerBeat = (config.gridResolution / 4).coerceAtLeast(1)
    private val beatsPerSecond = config.bpm / 60f
    private val windowBeats = (config.chordWindowSeconds * beatsPerSecond).toInt().coerceAtLeast(2)
    private val strideBeats = (config.chordStrideSeconds * beatsPerSecond).toInt().coerceAtLeast(1)
    private val maxBufferFrames = (windowBeats * stepsPerBeat * 2).coerceAtLeast(128)

    private val frameBuffer = FrameRingBuffer(maxBufferFrames)
    private var nextSampleBeat = 0L
    private var lastWindowChords: List<Int> = emptyList()
    private val stableChordTimeline = mutableListOf<Int>()

    fun ingest(frame: MelodyFrame): ChordUpdate? {
        frameBuffer.add(frame)
        val currentBeat = frame.stepIndex / stepsPerBeat
        if (currentBeat < nextSampleBeat) return null

        val windowFramesNeeded = windowBeats * stepsPerBeat
        val windowFrames = frameBuffer.latest(windowFramesNeeded)
        if (windowFrames.isEmpty()) return null

        val melody = LongArray(inferenceEngine.maxSeqLen) { 14L }
        val beatPositions = LongArray(inferenceEngine.maxSeqLen) { 0L }
        val actualLen = windowFrames.size.coerceAtMost(inferenceEngine.maxSeqLen)
        val start = (windowFrames.size - actualLen).coerceAtLeast(0)
        for (i in 0 until actualLen) {
            val f = windowFrames[start + i]
            melody[i] = f.melodyToken.toLong()
            beatPositions[i] = (f.stepIndex % config.gridResolution).toLong()
        }

        val windowChordIds = inferenceEngine.predict(melody, beatPositions, actualLen, stepsPerBeat)
        val merged = mergeOverlap(lastWindowChords, windowChordIds)
        appendStableChords(merged)
        val lagged = currentLaggedChords()

        onInferenceDebug?.invoke(
            InferenceDebugSnapshot(
                melodyTokens = melody.take(actualLen).map { it.toInt() },
                beatPositions = beatPositions.take(actualLen).map { it.toInt() },
                windowChordIds = windowChordIds,
                mergedWindowChords = merged,
                laggedChords = lagged,
            )
        )

        lastWindowChords = windowChordIds
        nextSampleBeat = currentBeat + strideBeats

        return ChordUpdate(
            rawWindowChords = windowChordIds,
            stableTimeline = stableChordTimeline.toList(),
            laggedChords = lagged,
        )
    }

    fun currentLaggedChords(): List<Int> {
        if (stableChordTimeline.isEmpty()) return emptyList()
        val lag = config.chordLagBeats.coerceAtLeast(0)
        val endExclusive = (stableChordTimeline.size - lag).coerceAtLeast(0)
        if (endExclusive <= 0) return emptyList()
        return stableChordTimeline.subList(0, endExclusive)
    }

    private fun appendStableChords(chords: List<Int>) {
        if (chords.isEmpty()) return
        if (stableChordTimeline.isEmpty()) {
            stableChordTimeline.addAll(chords)
            return
        }
        val overlap = (windowBeats - strideBeats).coerceAtLeast(0)
        if (overlap == 0) {
            stableChordTimeline.addAll(chords)
            return
        }
        val appendFrom = overlap.coerceAtMost(chords.size)
        stableChordTimeline.addAll(chords.subList(appendFrom, chords.size))
    }

    private fun mergeOverlap(previous: List<Int>, current: List<Int>): List<Int> {
        if (previous.isEmpty() || current.isEmpty()) return current
        val overlap = (windowBeats - strideBeats).coerceAtLeast(0).coerceAtMost(minOf(previous.size, current.size))
        if (overlap == 0) return current

        val merged = current.toMutableList()
        for (i in 0 until overlap) {
            val prevCid = previous[previous.size - overlap + i]
            val curCid = current[i]
            merged[i] = if (prevCid == curCid) curCid else prevCid
        }
        return merged
    }
}

data class InferenceDebugSnapshot(
    val melodyTokens: List<Int>,
    val beatPositions: List<Int>,
    val windowChordIds: List<Int>,
    val mergedWindowChords: List<Int>,
    val laggedChords: List<Int>,
)

data class ChordUpdate(
    val rawWindowChords: List<Int>,
    val stableTimeline: List<Int>,
    val laggedChords: List<Int>,
)

private class FrameRingBuffer(capacity: Int) {
    private val data = ArrayList<MelodyFrame>(capacity)
    private val maxCapacity = capacity

    fun add(frame: MelodyFrame) {
        if (data.size == maxCapacity) {
            data.removeAt(0)
        }
        data.add(frame)
    }

    fun latest(n: Int): List<MelodyFrame> {
        if (data.isEmpty()) return emptyList()
        val count = n.coerceAtMost(data.size)
        return data.subList(data.size - count, data.size)
    }
}
