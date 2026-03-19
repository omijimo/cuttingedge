package com.example.accompaniment.inference

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Runs the chord prediction TFLite model on-device.
 *
 * Input:  melody_tokens [1, seq_len] int64, beat_positions [1, seq_len] int64
 * Output: chord_logits  [1, seq_len, 86] float32
 */
class ChordInferenceEngine(context: Context, modelFilename: String = "chord_model.tflite") :
    Closeable {

    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context, modelFilename)
        val options = Interpreter.Options().apply {
            numThreads = 2
        }
        interpreter = Interpreter(model, options)
    }

    val maxSeqLen: Int
        get() = interpreter.getInputTensor(0).shape()[1]

    /**
     * Run inference and return per-beat chord IDs.
     *
     * @param melodyTokens  [maxSeqLen] melody token IDs
     * @param beatPositions [maxSeqLen] bar-position IDs
     * @param actualLen     number of real (non-pad) timesteps
     * @param stepsPerBeat  timesteps per beat (default 4 for 16th-note grid)
     * @return list of chord IDs, one per beat
     */
    fun predict(
        melodyTokens: LongArray,
        beatPositions: LongArray,
        actualLen: Int,
        stepsPerBeat: Int = 4,
    ): List<Int> {
        val seqLen = maxSeqLen
        require(melodyTokens.size == seqLen && beatPositions.size == seqLen)

        // Prepare inputs as [1, seqLen]
        val melInput = Array(1) { melodyTokens }
        val beatInput = Array(1) { beatPositions }

        // Output: [1, seqLen, CHORD_VOCAB_SIZE]
        val output = Array(1) { Array(seqLen) { FloatArray(ChordVocab.CHORD_VOCAB_SIZE) } }

        val inputs = arrayOf<Any>(melInput, beatInput)
        val outputs = mutableMapOf<Int, Any>(0 to output)
        interpreter.runForMultipleInputsOutputs(inputs, outputs)

        // Greedy decode: argmax per timestep, majority vote per beat
        val logits = output[0]
        val chordIds = mutableListOf<Int>()
        val len = actualLen.coerceAtMost(seqLen)

        var beatStart = 0
        while (beatStart < len) {
            val beatEnd = (beatStart + stepsPerBeat).coerceAtMost(len)
            val votes = mutableMapOf<Int, Int>()
            for (i in beatStart until beatEnd) {
                val pred = logits[i].indices.maxByOrNull { logits[i][it] } ?: ChordVocab.PAD_CHORD_ID
                if (pred != ChordVocab.PAD_CHORD_ID) {
                    votes[pred] = (votes[pred] ?: 0) + 1
                }
            }
            val winner = votes.maxByOrNull { it.value }?.key ?: ChordVocab.PAD_CHORD_ID
            chordIds.add(winner)
            beatStart = beatEnd
        }

        return chordIds
    }

    override fun close() {
        interpreter.close()
    }

    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        val input = FileInputStream(fd.fileDescriptor)
        val channel = input.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}
