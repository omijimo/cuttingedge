package com.example.accompaniment

import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.accompaniment.databinding.ActivityMainBinding
import com.example.accompaniment.generation.AccompanimentGenerator
import com.example.accompaniment.inference.ChordInferenceEngine
import com.example.accompaniment.inference.ChordVocab
import com.example.accompaniment.inference.MelodyTokenizer
import com.example.accompaniment.midi.MidiParser
import com.example.accompaniment.midi.MidiPlayer
import com.example.accompaniment.midi.MidiWriter
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var engine: ChordInferenceEngine
    private lateinit var player: MidiPlayer

    private var outputFile: File? = null
    private var sampleNames: List<String> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        engine = ChordInferenceEngine(this)
        player = MidiPlayer(this)

        // Discover sample MIDI files from assets/samples/
        sampleNames = assets.list("samples")
            ?.filter { it.endsWith(".mid") || it.endsWith(".midi") }
            ?.sorted()
            ?: emptyList()

        if (sampleNames.isEmpty()) {
            binding.statusText.text = "No sample MIDI files found in assets/samples/"
            binding.generateButton.isEnabled = false
            return
        }

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, sampleNames)
        binding.melodySpinner.adapter = adapter

        binding.generateButton.setOnClickListener { runPipeline() }
        binding.playButton.setOnClickListener { playOutput() }
        binding.stopButton.setOnClickListener { stopPlayback() }
    }

    private fun runPipeline() {
        val selectedName = sampleNames.getOrNull(binding.melodySpinner.selectedItemPosition) ?: return
        setUiRunning(true)

        lifecycleScope.launch {
            try {
                val (chordLabels, elapsedMs) = withContext(Dispatchers.Default) {
                    val t0 = System.nanoTime()

                    // 1. Parse MIDI
                    val midiData = assets.open("samples/$selectedName").use { MidiParser.parse(it) }

                    // 2. Tokenize
                    val tokenized = MelodyTokenizer.tokenize(
                        midiData,
                        gridResolution = 16,
                        maxSeqLen = engine.maxSeqLen,
                    )

                    // 3. Run TFLite inference
                    val chordIds = engine.predict(
                        tokenized.melodyTokens,
                        tokenized.beatPositions,
                        tokenized.numSteps,
                    )

                    // 4. Generate accompaniment
                    val accompNotes = AccompanimentGenerator.generate(
                        chordIds,
                        ticksPerBeat = midiData.ticksPerBeat,
                    )

                    // 5. Re-create melody notes as SimpleNotes for the output
                    val melodySimple = midiData.notes.map { n ->
                        MidiWriter.SimpleNote(
                            n.pitch, n.velocity, n.startTick,
                            (n.endTick - n.startTick).coerceAtLeast(1),
                        )
                    }

                    // 6. Write output MIDI
                    val outFile = File(cacheDir, "output_${selectedName}")
                    FileOutputStream(outFile).use { fos ->
                        MidiWriter.write(
                            fos, melodySimple, accompNotes,
                            ticksPerBeat = midiData.ticksPerBeat,
                            tempoMicros = midiData.tempoMicros,
                        )
                    }
                    outputFile = outFile

                    val labels = chordIds.map { ChordVocab.chordToString(it) }
                    val elapsed = (System.nanoTime() - t0) / 1_000_000
                    labels to elapsed
                }

                showResults(chordLabels, elapsedMs)
            } catch (e: Exception) {
                binding.statusText.text = getString(R.string.status_error, e.message)
                setUiRunning(false)
            }
        }
    }

    private fun showResults(chordLabels: List<String>, elapsedMs: Long) {
        binding.statusText.text = getString(R.string.status_done)
        binding.timingText.text = "Inference + generation: ${elapsedMs}ms"
        binding.timingText.visibility = View.VISIBLE
        binding.chordsLabel.visibility = View.VISIBLE
        binding.chordsText.text = chordLabels.chunked(8)
            .joinToString("\n") { it.joinToString("  ") }
        binding.chordsText.visibility = View.VISIBLE
        binding.playbackControls.visibility = View.VISIBLE
        setUiRunning(false)
    }

    private fun playOutput() {
        val file = outputFile ?: return
        player.play(file) {
            runOnUiThread { binding.statusText.text = "Playback finished" }
        }
        binding.statusText.text = "Playing…"
    }

    private fun stopPlayback() {
        player.stop()
        binding.statusText.text = getString(R.string.status_done)
    }

    private fun setUiRunning(running: Boolean) {
        binding.generateButton.isEnabled = !running
        binding.melodySpinner.isEnabled = !running
        if (running) {
            binding.statusText.text = getString(R.string.status_running)
            binding.playbackControls.visibility = View.GONE
            binding.chordsLabel.visibility = View.GONE
            binding.chordsText.visibility = View.GONE
            binding.timingText.visibility = View.GONE
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        player.close()
        engine.close()
    }
}
