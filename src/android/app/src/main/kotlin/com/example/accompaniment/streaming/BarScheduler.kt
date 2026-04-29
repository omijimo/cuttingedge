package com.example.accompaniment.streaming

import java.util.PriorityQueue

/**
 * Bar-aligned scheduler:
 * - queues generated accompaniment notes
 * - emits notes when their (lagged) playback tick is due
 *
 * scheduleBarLag controls how far behind live input we play generated output.
 */
class BarScheduler(
    private val config: StreamingConfig,
    private val ticksPerBeat: Int = 480,
) {
    private val ticksPerBar = ticksPerBeat * config.beatsPerBar
    private val lagTicks = ticksPerBar * config.scheduleBarLag.coerceAtLeast(0)

    private val queue = PriorityQueue<ScheduledNote>(compareBy { it.startTick })

    fun submit(notes: List<ScheduledNote>) {
        queue.addAll(notes)
    }

    /**
     * @param transportTick current live transport tick (input-side clock)
     * @return notes to play now (or earlier) after applying schedule lag
     */
    fun popDue(transportTick: Long): List<ScheduledNote> {
        val duePlaybackTick = (transportTick - lagTicks).coerceAtLeast(0)
        if (queue.isEmpty()) return emptyList()

        val out = mutableListOf<ScheduledNote>()
        while (queue.isNotEmpty()) {
            val next = queue.peek() ?: break
            if (next.startTick > duePlaybackTick) break
            out.add(queue.poll() ?: break)
        }
        return out
    }

    fun clear() {
        queue.clear()
    }
}
