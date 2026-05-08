package com.example.accompaniment

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.accompaniment.databinding.ActivityMainBinding
import com.example.accompaniment.generation.AccompanimentGenerator
import com.example.accompaniment.inference.ChordInferenceEngine
import com.example.accompaniment.inference.ChordVocab
import com.example.accompaniment.inference.MelodyTokenizer
import com.example.accompaniment.midi.MidiParser
import com.example.accompaniment.midi.MidiPlayer
import com.example.accompaniment.midi.RealtimeMidiPlayer
import com.example.accompaniment.midi.MidiWriter
import com.example.accompaniment.streaming.FrameState
import com.example.accompaniment.streaming.MelodyFrame
import com.example.accompaniment.streaming.RealTimeStreamingPipeline
import com.example.accompaniment.streaming.StreamingConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var engine: ChordInferenceEngine
    private lateinit var player: MidiPlayer
    private lateinit var realtimeTonePlayer: RealtimeMidiPlayer

    private var outputFile: File? = null
    private var sampleNames: List<String> = emptyList()
    private var isRealtimeRunning = false
    private var pendingRealtimeStart = false
    private var realtimeJob: Job? = null
    private var metronomeJob: Job? = null
    private var realtimePipeline: RealTimeStreamingPipeline? = null
    private var audioRecord: AudioRecord? = null
    @Volatile
    private var latestLaggedChords: List<Int> = emptyList()

    private val keyOptions = listOf("Auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    private val modeOptions = listOf("Auto", "Major", "Minor")
    private val quantOptions = listOf(8, 16)

    private companion object {
        const val REQUEST_RECORD_AUDIO = 4001
        const val TICKS_PER_BEAT = 480
        const val TAG_RT_DEBUG = "RT_DEBUG"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        engine = ChordInferenceEngine(this)
        player = MidiPlayer(this)
        realtimeTonePlayer = RealtimeMidiPlayer(this)

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
        setupRealtimeControls()

        binding.generateButton.setOnClickListener { runPipeline() }
        binding.playButton.setOnClickListener { playOutput() }
        binding.stopButton.setOnClickListener { stopPlayback() }
        binding.realtimePlayPauseButton.setOnClickListener { toggleRealtime() }
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
                    Log.d(
                        TAG_RT_DEBUG,
                        "offline.model_input_tokens=" +
                            tokenized.melodyTokens.take(tokenized.numSteps).joinToString(",")
                    )
                    Log.d(
                        TAG_RT_DEBUG,
                        "offline.model_input_positions=" +
                            tokenized.beatPositions.take(tokenized.numSteps).joinToString(",")
                    )
                    Log.d(
                        TAG_RT_DEBUG,
                        "offline.chord_output_tokens=${chordIds.joinToString(",")}"
                    )

                    // 4. Generate accompaniment
                    val accompNotes = AccompanimentGenerator.generate(
                        chordIds,
                        ticksPerBeat = midiData.ticksPerBeat,
                    )
                    Log.d(
                        TAG_RT_DEBUG,
                        "offline.accompaniment_output=" +
                            accompNotes.joinToString(";") {
                                "p=${it.pitch},t=${it.startTick},d=${it.durationTicks},v=${it.velocity}"
                            }
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

    private fun setupRealtimeControls() {
        binding.realtimeKeySpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            keyOptions,
        )
        binding.realtimeModeSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            modeOptions,
        )
        binding.realtimeQuantSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            quantOptions.map { "$it steps/bar" },
        )
        binding.realtimeQuantSpinner.setSelection(1)
    }

    private fun toggleRealtime() {
        if (isRealtimeRunning) {
            stopRealtime()
        } else {
            startRealtime()
        }
    }

    private fun startRealtime() {
        val useMidiDebugInput = binding.realtimeUseMidiSwitch.isChecked
        if (!useMidiDebugInput && !hasRecordPermission()) {
            pendingRealtimeStart = true
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO,
            )
            binding.realtimeStatusText.text = getString(R.string.realtime_status_permission)
            return
        }

        val config = buildRealtimeConfig()
        binding.realtimeStatusText.text = getString(R.string.realtime_status_starting)
        binding.realtimePlayPauseButton.text = getString(R.string.realtime_pause)
        isRealtimeRunning = true
        latestLaggedChords = emptyList()
        resetNowPlayingUi()

        realtimePipeline = RealTimeStreamingPipeline(
            config = config,
            chordInferenceEngine = engine,
            ticksPerBeat = TICKS_PER_BEAT,
            onInferenceDebug = { dbg ->
                latestLaggedChords = dbg.laggedChords
                Log.d(TAG_RT_DEBUG, "rt.model_input_tokens=${dbg.melodyTokens.joinToString(",")}")
                Log.d(TAG_RT_DEBUG, "rt.model_input_positions=${dbg.beatPositions.joinToString(",")}")
                Log.d(TAG_RT_DEBUG, "rt.chord_output_tokens=${dbg.windowChordIds.joinToString(",")}")
                Log.d(TAG_RT_DEBUG, "rt.chord_merged_window=${dbg.mergedWindowChords.joinToString(",")}")
                Log.d(TAG_RT_DEBUG, "rt.chord_lagged_timeline=${dbg.laggedChords.joinToString(",")}")
            },
            onPitchDebug = { p ->
                Log.d(
                    TAG_RT_DEBUG,
                    "rt.autocorr_frame step=${p.stepIndex},state=${p.state},midi=${p.midiPitch},conf=" +
                        "${"%.3f".format(p.confidence)},rms=${"%.4f".format(p.inputRms)},peak=${"%.4f".format(p.inputPeak)}"
                )
            },
            onAccompanimentDebug = { notes ->
                Log.d(
                    TAG_RT_DEBUG,
                    "rt.accompaniment_output=" +
                        notes.joinToString(";") {
                            "p=${it.pitch},t=${it.startTick},d=${it.durationTicks},v=${it.velocity}"
                        }
                )
            },
            onDueNotesDebug = { due ->
                Log.d(
                    TAG_RT_DEBUG,
                    "rt.scheduler_due_notes=" +
                        due.joinToString(";") {
                            "p=${it.pitch},t=${it.startTick},d=${it.durationTicks},v=${it.velocity}"
                        }
                )
            },
        )
        startMetronome(config)

        if (useMidiDebugInput) {
            startRealtimeFromSelectedMidi(config)
            return
        }
        startRealtimeFromMicrophone(config)
    }

    private fun startRealtimeFromMicrophone(config: StreamingConfig) {
        val minBuffer = AudioRecord.getMinBufferSize(
            config.sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        val frameSize = 1024
        val bufferSize = max(minBuffer, frameSize * 2)
        val recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            config.sampleRateHz,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
        )
        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            binding.realtimeStatusText.text = "AudioRecord failed to initialize"
            stopRealtime()
            return
        }
        audioRecord = recorder
        recorder.startRecording()

        val startNanos = System.nanoTime()
        binding.realtimeStatusText.text = getString(R.string.realtime_status_running)

        realtimeJob = lifecycleScope.launch(Dispatchers.Default) {
            val shortBuffer = ShortArray(frameSize)
            var totalSamples = 0L
            var lastUiUpdate = startNanos
            var generatedNotes = 0L

            try {
                while (isActive && isRealtimeRunning) {
                    val read = recorder.read(shortBuffer, 0, shortBuffer.size)
                    if (read <= 0) continue

                    val frame = FloatArray(read) { i -> shortBuffer[i] / 32768f }
                    val captureNanos = startNanos + ((totalSamples * 1_000_000_000L) / config.sampleRateHz)
                    totalSamples += read

                    val due = realtimePipeline?.onAudioFrame(frame, captureNanos).orEmpty()
                    if (due.isNotEmpty()) {
                        generatedNotes += due.size
                        realtimeTonePlayer.playNotes(due, config.bpm, TICKS_PER_BEAT)
                        updateNowPlayingUi(due)
                    }

                    if (captureNanos - lastUiUpdate > 1_000_000_000L) {
                        val notes = generatedNotes
                        runOnUiThread {
                            if (isRealtimeRunning) {
                                binding.realtimeStatusText.text =
                                    "Running — generated notes: $notes | Key: ${selectedKey()} | Mode: ${selectedMode()}"
                            }
                        }
                        lastUiUpdate = captureNanos
                    }
                }
            } catch (_: Exception) {
                runOnUiThread { binding.realtimeStatusText.text = "Real-time pipeline stopped due to runtime error" }
            }
        }
    }

    private fun startRealtimeFromSelectedMidi(config: StreamingConfig) {
        val selectedName = sampleNames.getOrNull(binding.melodySpinner.selectedItemPosition)
        if (selectedName == null) {
            binding.realtimeStatusText.text = "No selected MIDI file for debug input"
            stopRealtime()
            return
        }
        binding.realtimeStatusText.text = getString(R.string.realtime_status_running_midi)

        realtimeJob = lifecycleScope.launch(Dispatchers.Default) {
            try {
                val midiData = assets.open("samples/$selectedName").use { MidiParser.parse(it) }
                val tokenized = MelodyTokenizer.tokenize(
                    midiData,
                    gridResolution = config.gridResolution,
                    maxSeqLen = 4096,
                )

                val stepsPerBeat = (config.gridResolution / config.beatsPerBar).coerceAtLeast(1)
                val stepMs = (60_000f / config.bpm / stepsPerBeat).toLong().coerceAtLeast(10L)
                var generatedNotes = 0L
                var lastUiStep = 0

                for (i in 0 until tokenized.numSteps) {
                    if (!isActive || !isRealtimeRunning) break

                    val token = tokenized.melodyTokens[i].toInt()
                    val state = when (token) {
                        MelodyTokenizer.HOLD_TOKEN -> FrameState.HOLD
                        MelodyTokenizer.REST_TOKEN, MelodyTokenizer.PAD_TOKEN -> FrameState.REST
                        else -> FrameState.ON
                    }
                    val frame = MelodyFrame(
                        stepIndex = i.toLong(),
                        timestampNanos = System.nanoTime(),
                        midiPitch = if (state == FrameState.ON) token else -1,
                        state = state,
                        confidence = 1f,
                    )
                    val transportTick = (i.toLong() * TICKS_PER_BEAT) / stepsPerBeat
                    val due = realtimePipeline?.onMelodyFrame(frame, transportTick).orEmpty()
                    if (due.isNotEmpty()) {
                        generatedNotes += due.size
                        realtimeTonePlayer.playNotes(due, config.bpm, TICKS_PER_BEAT)
                        updateNowPlayingUi(due)
                    }

                    if (i - lastUiStep >= config.gridResolution) {
                        lastUiStep = i
                        val notes = generatedNotes
                        runOnUiThread {
                            if (isRealtimeRunning) {
                                binding.realtimeStatusText.text =
                                    "Running MIDI debug — generated notes: $notes | Source: $selectedName"
                            }
                        }
                    }
                    delay(stepMs)
                }

                // Flush a short tail so lagged scheduled notes can still play.
                val tailSteps = (config.scheduleBarLag + 1) * config.gridResolution
                val base = tokenized.numSteps
                for (j in 0 until tailSteps) {
                    if (!isActive || !isRealtimeRunning) break
                    val idx = base + j
                    val rest = MelodyFrame(
                        stepIndex = idx.toLong(),
                        timestampNanos = System.nanoTime(),
                        midiPitch = -1,
                        state = FrameState.REST,
                        confidence = 0f,
                    )
                    val transportTick = (idx.toLong() * TICKS_PER_BEAT) / stepsPerBeat
                    val due = realtimePipeline?.onMelodyFrame(rest, transportTick).orEmpty()
                    if (due.isNotEmpty()) {
                        realtimeTonePlayer.playNotes(due, config.bpm, TICKS_PER_BEAT)
                        updateNowPlayingUi(due)
                    }
                    delay(stepMs)
                }

                runOnUiThread {
                    if (isRealtimeRunning) {
                        binding.realtimeStatusText.text = "MIDI debug stream complete — tap Pause to stop or Play to rerun"
                    }
                }
            } catch (_: Exception) {
                runOnUiThread { binding.realtimeStatusText.text = "Failed to run MIDI debug stream" }
            }
        }
    }

    private fun stopRealtime() {
        isRealtimeRunning = false
        realtimeJob?.cancel()
        metronomeJob?.cancel()
        realtimeJob = null
        metronomeJob = null
        realtimeTonePlayer.stopPlayback()

        runCatching { audioRecord?.stop() }
        runCatching { audioRecord?.release() }
        audioRecord = null

        realtimePipeline?.close()
        realtimePipeline = null
        latestLaggedChords = emptyList()
        binding.realtimePlayPauseButton.text = getString(R.string.realtime_play)
        binding.realtimeStatusText.text = getString(R.string.realtime_status_paused)
        resetNowPlayingUi()
    }

    private fun startMetronome(config: StreamingConfig) {
        val beatMs = (60_000f / config.bpm).toLong().coerceAtLeast(60L)
        metronomeJob = lifecycleScope.launch {
            var beatInBar = 0
            while (isActive && isRealtimeRunning) {
                val accent = beatInBar == 0
                pulseRealtimeButton(accent)
                realtimeTonePlayer.playMetronome(accent)
                beatInBar = (beatInBar + 1) % config.beatsPerBar
                delay(beatMs)
            }
        }
    }

    private fun pulseRealtimeButton(accent: Boolean) {
        val targetScale = if (accent) 1.14f else 1.08f
        binding.realtimePlayPauseButton.animate()
            .scaleX(targetScale)
            .scaleY(targetScale)
            .setDuration(90)
            .withEndAction {
                binding.realtimePlayPauseButton.animate()
                    .scaleX(1f)
                    .scaleY(1f)
                    .setDuration(170)
                    .start()
            }
            .start()
    }

    private fun buildRealtimeConfig(): StreamingConfig {
        val bpm = binding.realtimeBpmInput.text?.toString()?.toFloatOrNull()?.coerceIn(40f, 240f) ?: 120f
        val quant = quantOptions.getOrElse(binding.realtimeQuantSpinner.selectedItemPosition) { 16 }
        val lagBars = binding.realtimeLagInput.text?.toString()?.toIntOrNull()?.coerceIn(1, 2) ?: 1
        return StreamingConfig(
            bpm = bpm,
            gridResolution = quant,
            scheduleBarLag = lagBars,
            chordLagBeats = 1,
            beatsPerBar = 4,
        )
    }

    private fun selectedKey(): String = keyOptions.getOrElse(binding.realtimeKeySpinner.selectedItemPosition) { "Auto" }

    private fun selectedMode(): String =
        modeOptions.getOrElse(binding.realtimeModeSpinner.selectedItemPosition) { "Auto" }

    private fun updateNowPlayingUi(due: List<com.example.accompaniment.streaming.ScheduledNote>) {
        for (note in due) {
            val chordLabel = chordLabelForTick(note.startTick)
            runOnUiThread {
                if (isRealtimeRunning) {
                    binding.realtimeNowPlayingChordText.text =
                        getString(R.string.realtime_now_playing_chord_value, chordLabel)
                }
            }
        }
    }

    private fun chordLabelForTick(startTick: Long): String {
        val lagged = latestLaggedChords
        if (lagged.isEmpty()) return "N.C."
        val beatIndex = (startTick / TICKS_PER_BEAT).toInt().coerceAtLeast(0)
        val chordId = lagged.getOrElse(beatIndex) { lagged.last() }
        return ChordVocab.chordToString(chordId)
    }

    private fun resetNowPlayingUi() {
        binding.realtimeNowPlayingChordText.text = getString(R.string.realtime_now_playing_chord_placeholder)
    }

    private fun hasRecordPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO) {
            val granted = grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED
            if (granted && pendingRealtimeStart) {
                pendingRealtimeStart = false
                startRealtime()
            } else {
                pendingRealtimeStart = false
                binding.realtimeStatusText.text = getString(R.string.realtime_status_permission)
            }
        }
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
        stopRealtime()
        if (::realtimeTonePlayer.isInitialized) realtimeTonePlayer.close()
        super.onDestroy()
        player.close()
        engine.close()
    }
}
