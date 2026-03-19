package com.example.accompaniment.midi

import java.io.ByteArrayOutputStream
import java.io.OutputStream

/**
 * Minimal MIDI file writer. Produces a Type 1 file with two tracks
 * (melody + accompaniment).
 */
object MidiWriter {

    data class SimpleNote(
        val pitch: Int,
        val velocity: Int,
        val startTick: Long,
        val durationTicks: Long,
    )

    fun write(
        output: OutputStream,
        melodyNotes: List<SimpleNote>,
        accompNotes: List<SimpleNote>,
        ticksPerBeat: Int = 480,
        tempoMicros: Long = 500_000, // 120 BPM
    ) {
        val buf = ByteArrayOutputStream()

        // Header: Format 1, 2 tracks
        buf.write("MThd".toByteArray())
        buf.writeInt(6)
        buf.writeShort(1) // format
        buf.writeShort(2) // 2 tracks
        buf.writeShort(ticksPerBeat)

        writeTrack(buf, melodyNotes, tempoMicros, includeTempo = true)
        writeTrack(buf, accompNotes, tempoMicros, includeTempo = false)

        output.write(buf.toByteArray())
    }

    private fun writeTrack(
        out: ByteArrayOutputStream,
        notes: List<SimpleNote>,
        tempoMicros: Long,
        includeTempo: Boolean,
    ) {
        val track = ByteArrayOutputStream()

        if (includeTempo) {
            // Tempo meta event at tick 0
            track.writeVarLen(0)
            track.write(0xFF)
            track.write(0x51)
            track.write(0x03)
            track.write(((tempoMicros shr 16) and 0xFF).toInt())
            track.write(((tempoMicros shr 8) and 0xFF).toInt())
            track.write((tempoMicros and 0xFF).toInt())
        }

        // Build event list sorted by time
        data class Event(val tick: Long, val status: Int, val d1: Int, val d2: Int)

        val events = mutableListOf<Event>()
        for (n in notes) {
            events.add(Event(n.startTick, 0x90, n.pitch, n.velocity))
            events.add(Event(n.startTick + n.durationTicks, 0x80, n.pitch, 0))
        }
        events.sortWith(compareBy({ it.tick }, { it.status })) // note-off before note-on at same tick

        var lastTick = 0L
        for (e in events) {
            track.writeVarLen(e.tick - lastTick)
            track.write(e.status)
            track.write(e.d1)
            track.write(e.d2)
            lastTick = e.tick
        }

        // End of track
        track.writeVarLen(0)
        track.write(0xFF)
        track.write(0x2F)
        track.write(0x00)

        val trackBytes = track.toByteArray()
        out.write("MTrk".toByteArray())
        out.writeInt(trackBytes.size)
        out.write(trackBytes)
    }

    private fun ByteArrayOutputStream.writeInt(v: Int) {
        write((v shr 24) and 0xFF)
        write((v shr 16) and 0xFF)
        write((v shr 8) and 0xFF)
        write(v and 0xFF)
    }

    private fun ByteArrayOutputStream.writeShort(v: Int) {
        write((v shr 8) and 0xFF)
        write(v and 0xFF)
    }

    private fun ByteArrayOutputStream.writeVarLen(value: Long) {
        if (value < 0x80) {
            write(value.toInt())
            return
        }
        val buf = mutableListOf<Int>()
        var v = value
        buf.add((v and 0x7F).toInt())
        v = v shr 7
        while (v > 0) {
            buf.add(((v and 0x7F) or 0x80).toInt())
            v = v shr 7
        }
        for (b in buf.reversed()) write(b)
    }
}
