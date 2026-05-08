package com.example.accompaniment.midi

import android.content.Context
import android.media.AudioManager
import android.media.MediaPlayer
import android.media.ToneGenerator
import com.example.accompaniment.streaming.ScheduledNote
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.util.ArrayDeque
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.roundToLong

/**
 * Real-time accompaniment playback that renders due note batches as small MIDI clips.
 *
 * This uses Android's platform MIDI synthesizer via MediaPlayer so we can play
 * model MIDI output directly, instead of generating custom PCM sine tones.
 */
class RealtimeMidiPlayer(
    context: Context,
) : Closeable {
    private val clipDir = File(context.cacheDir, "rt_midi_clips").apply { mkdirs() }
    private val renderExecutor = Executors.newSingleThreadScheduledExecutor()
    private val toneGenerator = ToneGenerator(AudioManager.STREAM_MUSIC, 24)
    private val activePlayers = ArrayDeque<MediaPlayer>()
    private val pendingNotes = mutableListOf<ScheduledNote>()
    private val lock = Any()
    private var clipCounter = 0L
    private var released = false
    private var flushScheduled = false
    private var pendingBpm = 120f
    private var pendingTicksPerBeat = 480
    private var playbackGeneration = 0L

    private companion object {
        const val MAX_CONCURRENT_CLIPS = 3
        const val BATCH_WINDOW_MS = 22L
    }

    fun playMetronome(accent: Boolean) {
        val tone = if (accent) ToneGenerator.TONE_PROP_BEEP2 else ToneGenerator.TONE_PROP_BEEP
        toneGenerator.startTone(tone, 40)
    }

    fun playNotes(notes: List<ScheduledNote>, bpm: Float, ticksPerBeat: Int) {
        if (released || notes.isEmpty() || bpm <= 0f || ticksPerBeat <= 0) return
        synchronized(lock) {
            pendingNotes.addAll(notes)
            pendingBpm = bpm
            pendingTicksPerBeat = ticksPerBeat
            if (flushScheduled) return
            flushScheduled = true
        }
        // Small batching window merges adjacent scheduler bursts to reduce clip churn.
        renderExecutor.schedule({ flushPendingBatch() }, BATCH_WINDOW_MS, TimeUnit.MILLISECONDS)
    }

    /**
     * Hard-stop live playback without releasing this instance, so it can be reused
     * on the next start of the real-time stream.
     */
    fun stopPlayback() {
        val toStop = mutableListOf<MediaPlayer>()
        synchronized(lock) {
            playbackGeneration++
            pendingNotes.clear()
            flushScheduled = false
            while (activePlayers.isNotEmpty()) {
                toStop.add(activePlayers.removeFirst())
            }
        }
        toneGenerator.stopTone()
        for (player in toStop) {
            runCatching {
                if (player.isPlaying) player.stop()
                player.release()
            }
        }
        clipDir.listFiles()?.forEach { file ->
            if (file.extension.equals("mid", ignoreCase = true)) {
                runCatching { file.delete() }
            }
        }
    }

    override fun close() = release()

    fun release() {
        synchronized(lock) {
            if (released) return
            released = true
            playbackGeneration++
            pendingNotes.clear()
            flushScheduled = false
        }
        runCatching { renderExecutor.shutdownNow() }
        synchronized(lock) {
            while (activePlayers.isNotEmpty()) {
                val p = activePlayers.removeFirst()
                runCatching {
                    if (p.isPlaying) p.stop()
                    p.release()
                }
            }
        }
        toneGenerator.release()
    }

    private fun flushPendingBatch() {
        val batch: List<ScheduledNote>
        val bpm: Float
        val ticksPerBeat: Int
        val generation: Long
        synchronized(lock) {
            flushScheduled = false
            if (released || pendingNotes.isEmpty()) return
            batch = pendingNotes.toList()
            pendingNotes.clear()
            bpm = pendingBpm
            ticksPerBeat = pendingTicksPerBeat
            generation = playbackGeneration
        }
        renderAndPlay(batch, bpm, ticksPerBeat, generation)
    }

    private fun renderAndPlay(notes: List<ScheduledNote>, bpm: Float, ticksPerBeat: Int, generation: Long) {
        val normalized = normalizeNotes(notes) ?: return
        synchronized(lock) {
            if (released || generation != playbackGeneration) return
        }
        val tempoMicros = (60_000_000.0 / bpm.toDouble()).roundToLong().coerceAtLeast(1L)
        val clipFile = File(clipDir, "clip_${System.nanoTime()}_${clipCounter++}.mid")
        try {
            FileOutputStream(clipFile).use { out ->
                MidiWriter.write(
                    output = out,
                    melodyNotes = emptyList(),
                    accompNotes = normalized,
                    ticksPerBeat = ticksPerBeat,
                    tempoMicros = tempoMicros,
                )
            }
            startClipPlayback(clipFile, generation)
        } catch (_: Exception) {
            runCatching { clipFile.delete() }
        }
    }

    private fun normalizeNotes(notes: List<ScheduledNote>): List<MidiWriter.SimpleNote>? {
        if (notes.isEmpty()) return null
        val minTick = notes.minOf { it.startTick }
        return notes.mapNotNull { note ->
            val duration = note.durationTicks.coerceAtLeast(1L)
            val start = (note.startTick - minTick).coerceAtLeast(0L)
            val velocity = note.velocity.coerceIn(1, 127)
            val pitch = note.pitch.coerceIn(0, 127)
            MidiWriter.SimpleNote(
                pitch = pitch,
                velocity = velocity,
                startTick = start,
                durationTicks = duration,
            )
        }
    }

    private fun startClipPlayback(clipFile: File, generation: Long) {
        val player = MediaPlayer()
        try {
            synchronized(lock) {
                if (released || generation != playbackGeneration) {
                    runCatching { clipFile.delete() }
                    return
                }
                while (activePlayers.size >= MAX_CONCURRENT_CLIPS) {
                    val old = activePlayers.removeFirst()
                    runCatching {
                        if (old.isPlaying) old.stop()
                        old.release()
                    }
                }
                activePlayers.addLast(player)
            }
            player.setDataSource(clipFile.absolutePath)
            player.setOnCompletionListener { completed ->
                cleanupPlayer(completed, clipFile)
            }
            player.setOnErrorListener { errored, _, _ ->
                cleanupPlayer(errored, clipFile)
                true
            }
            player.prepare()
            player.start()
        } catch (_: Exception) {
            cleanupPlayer(player, clipFile)
        }
    }

    private fun cleanupPlayer(player: MediaPlayer, clipFile: File) {
        synchronized(lock) {
            activePlayers.remove(player)
        }
        runCatching {
            if (player.isPlaying) player.stop()
        }
        runCatching { player.release() }
        runCatching { clipFile.delete() }
    }
}
