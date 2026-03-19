package com.example.accompaniment.generation

import com.example.accompaniment.inference.ChordVocab
import com.example.accompaniment.midi.MidiWriter

/**
 * Generates accompaniment MIDI notes from a chord sequence.
 * Uses a simple block-chord + bass pattern.
 */
object AccompanimentGenerator {

    /**
     * @param chordIds     one chord ID per beat
     * @param ticksPerBeat MIDI ticks per beat
     * @param octave       base octave for accompaniment voicings
     * @param velocity     note velocity (0-127)
     * @return list of SimpleNote for the accompaniment track
     */
    fun generate(
        chordIds: List<Int>,
        ticksPerBeat: Int = 480,
        octave: Int = 3,
        velocity: Int = 70,
    ): List<MidiWriter.SimpleNote> {
        val notes = mutableListOf<MidiWriter.SimpleNote>()
        var prevPitches: IntArray? = null

        var i = 0
        while (i < chordIds.size) {
            val cid = chordIds[i]

            // Count run of identical chords
            var run = 1
            while (i + run < chordIds.size && chordIds[i + run] == cid) run++

            val (root, quality) = ChordVocab.decodeChord(cid)
            val totalTicks = run.toLong() * ticksPerBeat
            val startTick = i.toLong() * ticksPerBeat

            if (cid != ChordVocab.NO_CHORD_ID && cid != ChordVocab.PAD_CHORD_ID && root != "N.C.") {
                var pitches = ChordVocab.chordToPitches(root, quality, octave)
                if (prevPitches != null && pitches.isNotEmpty()) {
                    pitches = voiceLead(prevPitches, pitches)
                }
                prevPitches = pitches

                if (pitches.isNotEmpty()) {
                    val bass = pitches.min()
                    val upper = pitches.filter { it != bass }.ifEmpty { pitches.toList() }

                    // Bass on each beat, upper voices on off-beats (simple bass+chord pattern)
                    for (b in 0 until run) {
                        val t = startTick + b * ticksPerBeat
                        val dur = (ticksPerBeat * 0.9).toLong()
                        if (b % 2 == 0) {
                            notes.add(MidiWriter.SimpleNote(bass, velocity, t, dur))
                        } else {
                            for (p in upper) {
                                notes.add(MidiWriter.SimpleNote(p, velocity - 10, t, dur))
                            }
                        }
                    }
                }
            }
            i += run
        }
        return notes
    }

    /** Minimal-motion voice leading. */
    private fun voiceLead(prev: IntArray, target: IntArray): IntArray {
        if (prev.isEmpty()) return target
        val ref = prev.average()
        return target.map { t ->
            val candidates = (-2..2).map { t + it * 12 }
            candidates.minByOrNull { kotlin.math.abs(it - ref) }
                ?.coerceIn(36, 72) ?: t
        }.sorted().toIntArray()
    }
}
