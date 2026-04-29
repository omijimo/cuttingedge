package com.example.accompaniment.streaming

/**
 * Consumes lagged chord timeline and emits accompaniment notes from a recent
 * sliding chord window. This keeps generation stable while remaining reactive.
 */
class SlidingAccompanimentEngine(
    private val config: StreamingConfig,
    private val nativeEngine: NativeAccompanimentEngine = NativeAccompanimentEngine(),
) {
    private var generatedBeatCount = 0

    fun generateIncremental(chordsPerBeat: List<Int>, ticksPerBeat: Int = 480): List<ScheduledNote> {
        if (chordsPerBeat.isEmpty()) return emptyList()

        val endBeat = chordsPerBeat.size
        if (endBeat <= generatedBeatCount) return emptyList()

        val window = config.accompanimentWindowChords.coerceAtLeast(1)
        val startBeat = (endBeat - window).coerceAtLeast(generatedBeatCount)
        if (startBeat >= endBeat) return emptyList()

        val slice = chordsPerBeat.subList(startBeat, endBeat)
        val notes = nativeEngine.generate(
            chordIds = slice,
            startBeat = startBeat,
            ticksPerBeat = ticksPerBeat,
        )
        generatedBeatCount = endBeat
        return notes
    }

    fun reset() {
        generatedBeatCount = 0
    }
}
