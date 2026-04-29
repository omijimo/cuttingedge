package com.example.accompaniment.streaming

import com.example.accompaniment.inference.ChordInferenceEngine

/**
 * High-level orchestrator for streaming:
 * audio frame -> melody frame -> chord windows -> accompaniment notes -> scheduler.
 *
 * This class is UI-agnostic and can be wired into an AudioRecord + synth pipeline later.
 */
class RealTimeStreamingPipeline(
    config: StreamingConfig,
    chordInferenceEngine: ChordInferenceEngine,
    private val ticksPerBeat: Int = 480,
    onInferenceDebug: ((InferenceDebugSnapshot) -> Unit)? = null,
    private val onAccompanimentDebug: ((List<ScheduledNote>) -> Unit)? = null,
    private val onDueNotesDebug: ((List<ScheduledNote>) -> Unit)? = null,
) : AutoCloseable {

    private val pitchDetector = NativePitchDetector(config)
    private val chordEngine = StreamingChordEngine(config, chordInferenceEngine, onInferenceDebug)
    private val accompanimentEngine = SlidingAccompanimentEngine(config)
    private val scheduler = BarScheduler(config, ticksPerBeat)

    private val nanosPerBeat = (60_000_000_000.0 / config.bpm).toLong()

    /**
     * Process one PCM frame and return any accompaniment notes due for playback now.
     */
    fun onAudioFrame(pcm: FloatArray, captureTimeNanos: Long): List<ScheduledNote> {
        val melodyFrame = pitchDetector.processFrame(pcm, captureTimeNanos)
        val transportBeat = captureTimeNanos / nanosPerBeat
        val transportTick = transportBeat * ticksPerBeat
        return onMelodyFrame(melodyFrame, transportTick)
    }

    /**
     * Process a precomputed melody frame (e.g. from MIDI debug playback) through
     * the same streaming chord/accompaniment scheduler path.
     */
    fun onMelodyFrame(melodyFrame: MelodyFrame, transportTick: Long): List<ScheduledNote> {
        val update = chordEngine.ingest(melodyFrame)
        if (update != null) {
            val notes = accompanimentEngine.generateIncremental(update.laggedChords, ticksPerBeat)
            scheduler.submit(notes)
            if (notes.isNotEmpty()) {
                onAccompanimentDebug?.invoke(notes)
            }
        }
        val due = scheduler.popDue(transportTick)
        if (due.isNotEmpty()) {
            onDueNotesDebug?.invoke(due)
        }
        return due
    }

    fun reset() {
        pitchDetector.reset()
        accompanimentEngine.reset()
        scheduler.clear()
    }

    override fun close() {
        pitchDetector.close()
        reset()
    }
}
