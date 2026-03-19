package com.example.accompaniment.inference

import com.example.accompaniment.midi.MidiParser

/**
 * Tokenizes MIDI note events into melody tokens + beat positions,
 * matching the Python tokenization exactly.
 *
 * Vocabulary: 0-11 = pitch class, 12 = HOLD, 13 = REST, 14 = PAD
 * Beat positions: 0-15 (16th-note position within a bar in 4/4)
 */
object MelodyTokenizer {

    const val HOLD_TOKEN = 12
    const val REST_TOKEN = 13
    const val PAD_TOKEN = 14
    const val MELODY_VOCAB_SIZE = 15
    const val MAX_BAR_POSITIONS = 16

    data class TokenizedMelody(
        val melodyTokens: LongArray,
        val beatPositions: LongArray,
        val numSteps: Int,
    )

    fun tokenize(
        midi: MidiParser.MidiData,
        gridResolution: Int = 16,
        maxSeqLen: Int = 128,
    ): TokenizedMelody {
        val ticksPerBeat = midi.ticksPerBeat
        val beatsPerBar = midi.numerator
        val stepsPerBeat = gridResolution / beatsPerBar
        val ticksPerStep = ticksPerBeat.toDouble() / stepsPerBeat

        if (midi.notes.isEmpty()) {
            return TokenizedMelody(
                LongArray(maxSeqLen) { PAD_TOKEN.toLong() },
                LongArray(maxSeqLen) { 0L },
                0,
            )
        }

        val lastTick = midi.notes.maxOf { it.endTick }
        val numSteps = ((lastTick / ticksPerStep).toInt() + 1).coerceAtMost(maxSeqLen)

        // Quantize notes onto the grid, prefer higher pitch at each step
        val pitchGrid = IntArray(numSteps) { -1 }
        val stateGrid = IntArray(numSteps) { REST_TOKEN }

        val sortedNotes = midi.notes.sortedWith(compareBy({ it.startTick }, { -it.pitch }))

        for (note in sortedNotes) {
            val startStep = ((note.startTick / ticksPerStep).toInt()).coerceIn(0, numSteps - 1)
            val endStep = ((note.endTick / ticksPerStep).toInt()).coerceIn(startStep + 1, numSteps)

            if (pitchGrid[startStep] == -1 || note.pitch > pitchGrid[startStep]) {
                pitchGrid[startStep] = note.pitch
                stateGrid[startStep] = note.pitch % 12 // pitch class token
            }
            for (s in (startStep + 1) until endStep) {
                if (pitchGrid[s] == -1) {
                    pitchGrid[s] = note.pitch
                    stateGrid[s] = HOLD_TOKEN
                }
            }
        }

        // Build padded output arrays
        val melodyTokens = LongArray(maxSeqLen) { PAD_TOKEN.toLong() }
        val beatPositions = LongArray(maxSeqLen) { 0L }

        for (i in 0 until numSteps) {
            melodyTokens[i] = stateGrid[i].toLong()
            beatPositions[i] = (i % gridResolution).toLong()
        }

        return TokenizedMelody(melodyTokens, beatPositions, numSteps)
    }
}
