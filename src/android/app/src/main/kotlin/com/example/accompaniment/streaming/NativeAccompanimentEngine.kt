package com.example.accompaniment.streaming

/**
 * Native accompaniment generator for low-latency streaming note generation.
 *
 * Input chord IDs are interpreted as one chord per beat.
 * Output is a list of MIDI-note events in ticks.
 */
class NativeAccompanimentEngine {

    fun generate(
        chordIds: List<Int>,
        startBeat: Int,
        ticksPerBeat: Int = 480,
        octave: Int = 3,
        velocity: Int = 70,
    ): List<ScheduledNote> {
        if (chordIds.isEmpty()) return emptyList()
        val flat = nativeGenerate(
            chordIds.toIntArray(),
            startBeat,
            ticksPerBeat,
            octave,
            velocity,
        )
        if (flat.isEmpty()) return emptyList()

        val out = ArrayList<ScheduledNote>(flat.size / 4)
        var i = 0
        while (i + 3 < flat.size) {
            out.add(
                ScheduledNote(
                    pitch = flat[i],
                    startTick = flat[i + 1].toLong(),
                    durationTicks = flat[i + 2].toLong(),
                    velocity = flat[i + 3],
                )
            )
            i += 4
        }
        return out
    }

    private external fun nativeGenerate(
        chordIds: IntArray,
        startBeat: Int,
        ticksPerBeat: Int,
        octave: Int,
        velocity: Int,
    ): IntArray

    companion object {
        init {
            System.loadLibrary("accompaniment_streaming_native")
        }
    }
}
