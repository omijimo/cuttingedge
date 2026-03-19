package com.example.accompaniment.inference

/**
 * Chord vocabulary matching the Python training code.
 * 12 roots x 7 qualities + no-chord + pad = 86 tokens.
 */
object ChordVocab {
    val ROOTS = arrayOf("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    const val NUM_ROOTS = 12

    val QUALITIES = arrayOf("maj", "min", "dim", "aug", "dom7", "maj7", "min7")
    const val NUM_QUALITIES = 7

    const val NO_CHORD_ID = NUM_ROOTS * NUM_QUALITIES   // 84
    const val PAD_CHORD_ID = NO_CHORD_ID + 1             // 85
    const val CHORD_VOCAB_SIZE = PAD_CHORD_ID + 1        // 86

    // Intervals for generating MIDI pitches
    private val INTERVALS = mapOf(
        "maj" to intArrayOf(0, 4, 7),
        "min" to intArrayOf(0, 3, 7),
        "dim" to intArrayOf(0, 3, 6),
        "aug" to intArrayOf(0, 4, 8),
        "dom7" to intArrayOf(0, 4, 7, 10),
        "maj7" to intArrayOf(0, 4, 7, 11),
        "min7" to intArrayOf(0, 3, 7, 10),
    )

    fun decodeChord(chordId: Int): Pair<String, String> = when {
        chordId == NO_CHORD_ID -> "N.C." to ""
        chordId == PAD_CHORD_ID -> "PAD" to ""
        chordId < 0 || chordId >= NO_CHORD_ID -> "N.C." to ""
        else -> ROOTS[chordId / NUM_QUALITIES] to QUALITIES[chordId % NUM_QUALITIES]
    }

    fun chordToString(chordId: Int): String {
        val (root, quality) = decodeChord(chordId)
        return if (root == "N.C." || root == "PAD") root else "$root$quality"
    }

    fun chordToPitches(root: String, quality: String, octave: Int = 4): IntArray {
        if (root == "N.C." || root == "PAD" || root.isEmpty()) return intArrayOf()
        val rootIdx = ROOTS.indexOf(root)
        if (rootIdx < 0) return intArrayOf()
        val base = rootIdx + octave * 12
        val intervals = INTERVALS[quality] ?: intArrayOf(0, 4, 7)
        return intervals.map { base + it }.toIntArray()
    }
}
