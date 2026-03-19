package com.example.accompaniment.midi

import java.io.InputStream

/**
 * Minimal Standard MIDI File parser.
 * Extracts note events, tempo, and time signature from a Type 0 or Type 1 MIDI file.
 */
object MidiParser {

    data class NoteEvent(
        val pitch: Int,
        val velocity: Int,
        val startTick: Long,
        val endTick: Long,
    )

    data class MidiData(
        val notes: List<NoteEvent>,
        val ticksPerBeat: Int,
        val tempoMicros: Long, // microseconds per beat
        val numerator: Int,
        val denominator: Int,
    ) {
        val tempoBpm: Double get() = 60_000_000.0 / tempoMicros
    }

    fun parse(input: InputStream): MidiData {
        val bytes = input.readBytes()
        var pos = 0

        fun readByte(): Int = bytes[pos++].toInt() and 0xFF
        fun readShort(): Int = (readByte() shl 8) or readByte()
        fun readInt(): Int = (readByte() shl 24) or (readByte() shl 16) or (readByte() shl 8) or readByte()
        fun readString(n: Int): String = String(bytes, pos, n, Charsets.US_ASCII).also { pos += n }

        fun readVarLen(): Long {
            var value = 0L
            while (true) {
                val b = readByte()
                value = (value shl 7) or (b.toLong() and 0x7F)
                if (b and 0x80 == 0) break
            }
            return value
        }

        // Header chunk
        val headerTag = readString(4)
        require(headerTag == "MThd") { "Not a MIDI file" }
        val headerLen = readInt()
        val format = readShort()
        val numTracks = readShort()
        val division = readShort()
        if (headerLen > 6) pos += headerLen - 6

        val ticksPerBeat = division // assuming ticks-per-beat (not SMPTE)
        var tempoMicros = 500_000L // default 120 BPM
        var numerator = 4
        var denominator = 4

        data class PendingNote(val pitch: Int, val velocity: Int, val startTick: Long)

        val allNotes = mutableListOf<NoteEvent>()

        for (t in 0 until numTracks) {
            val trackTag = readString(4)
            val trackLen = readInt()
            if (trackTag != "MTrk") { pos += trackLen; continue }

            val trackEnd = pos + trackLen
            var tick = 0L
            var runningStatus = 0
            val pending = mutableMapOf<Int, PendingNote>()

            while (pos < trackEnd) {
                val delta = readVarLen()
                tick += delta

                var status = readByte()
                if (status < 0x80) {
                    // Running status: reuse previous, byte is data
                    pos--
                    status = runningStatus
                } else {
                    runningStatus = status
                }

                val type = status and 0xF0
                when {
                    type == 0x90 -> { // Note On
                        val pitch = readByte()
                        val vel = readByte()
                        if (vel > 0) {
                            pending[pitch] = PendingNote(pitch, vel, tick)
                        } else {
                            pending.remove(pitch)?.let {
                                allNotes.add(NoteEvent(it.pitch, it.velocity, it.startTick, tick))
                            }
                        }
                    }
                    type == 0x80 -> { // Note Off
                        val pitch = readByte()
                        readByte() // velocity
                        pending.remove(pitch)?.let {
                            allNotes.add(NoteEvent(it.pitch, it.velocity, it.startTick, tick))
                        }
                    }
                    type == 0xA0 -> { readByte(); readByte() } // Aftertouch
                    type == 0xB0 -> { readByte(); readByte() } // Control Change
                    type == 0xC0 -> { readByte() }             // Program Change
                    type == 0xD0 -> { readByte() }             // Channel Pressure
                    type == 0xE0 -> { readByte(); readByte() } // Pitch Bend
                    status == 0xFF -> { // Meta event
                        val metaType = readByte()
                        val len = readVarLen().toInt()
                        val metaStart = pos
                        when (metaType) {
                            0x51 -> { // Tempo
                                if (len >= 3) {
                                    tempoMicros = ((readByte().toLong() shl 16)
                                            or (readByte().toLong() shl 8)
                                            or readByte().toLong())
                                }
                            }
                            0x58 -> { // Time Signature
                                if (len >= 2) {
                                    numerator = readByte()
                                    val denomPow = readByte()
                                    denominator = 1 shl denomPow
                                }
                            }
                        }
                        pos = metaStart + len
                    }
                    status == 0xF0 || status == 0xF7 -> { // SysEx
                        val len = readVarLen().toInt()
                        pos += len
                    }
                    else -> {} // Unknown, skip
                }
            }

            // Close any pending notes at track end
            for ((_, pn) in pending) {
                allNotes.add(NoteEvent(pn.pitch, pn.velocity, pn.startTick, tick))
            }
        }

        allNotes.sortBy { it.startTick }
        return MidiData(allNotes, ticksPerBeat, tempoMicros, numerator, denominator)
    }
}
