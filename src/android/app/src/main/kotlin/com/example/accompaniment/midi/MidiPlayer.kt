package com.example.accompaniment.midi

import android.content.Context
import android.media.MediaPlayer
import java.io.Closeable
import java.io.File

/**
 * Simple MIDI playback using Android's MediaPlayer.
 * MediaPlayer delegates to the platform's Sonivox synthesizer for .mid files.
 */
class MidiPlayer(private val context: Context) : Closeable {

    private var player: MediaPlayer? = null

    val isPlaying: Boolean get() = player?.isPlaying == true

    fun play(midiFile: File, onCompletion: (() -> Unit)? = null) {
        stop()
        player = MediaPlayer().apply {
            setDataSource(midiFile.absolutePath)
            setOnCompletionListener { onCompletion?.invoke() }
            prepare()
            start()
        }
    }

    fun stop() {
        player?.let {
            if (it.isPlaying) it.stop()
            it.release()
        }
        player = null
    }

    override fun close() = stop()
}
