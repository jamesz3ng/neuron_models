"""
Memory Benchmark for Fast Model Delay Implementations

Compares memory usage and performance of different delay buffer strategies:
- Method A: Dense Ring Buffer (stores voltage for all neurons × delay_steps)
- Method B: Sparse Event Queue (stores only spike times)
- Method C: NumPy Boolean Buffer (1 byte per entry)
- Method D: Current fast_model implementation (full waveform storage)

This benchmark simulates real-time streaming scenarios where we can't
store the entire waveform history.

Simulation Parameters:
- N_neurons: 1,000 to 100,000
- delay_steps: 500 (5ms delay at 0.01ms dt)
- steps: 10,000 (100ms simulation)
- firing_rate_prob: 0.001 per step (~100 Hz effective rate)
"""

import time
import sys
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from src.fast_model import simulate_fast_model


# =============================================================================
# Configuration
# =============================================================================

N_NEURON_COUNTS = [1_000, 5_000, 10_000, 50_000, 100_000]
DELAY_STEPS = 500  # 5ms at 0.01ms dt
SIMULATION_STEPS = 10_000  # 100ms simulation
FIRING_PROB = 0.001  # ~100 Hz per neuron
SPIKE_VOLTAGE = 30.0  # mV
REST_VOLTAGE = -65.0  # mV
TIMEOUT_SECONDS = 30.0  # Skip if too slow
DT_MS = 0.01  # Time step in ms
DELAY_MS = DELAY_STEPS * DT_MS  # 5ms delay


# =============================================================================
# Method A: Dense Ring Buffer
# =============================================================================


def _benchmark_dense_ring_buffer(n_neurons: int) -> dict:
    """
    Dense ring buffer approach.

    Memory: O(N_neurons × delay_steps) float64
    Each neuron has a circular buffer storing voltage history.
    """
    # Allocate buffer: (N_neurons, delay_steps) float64
    buffer = np.full((n_neurons, DELAY_STEPS), REST_VOLTAGE, dtype=np.float64)

    # Track memory
    buffer_bytes = buffer.nbytes

    # Output accumulator (just to prevent optimization)
    total_output_spikes = 0

    start = time.perf_counter()

    for t in range(SIMULATION_STEPS):
        # Generate random spikes (vectorized)
        spikes = np.random.random(n_neurons) < FIRING_PROB

        # Current write position in ring buffer
        write_idx = t % DELAY_STEPS

        # Write inputs to buffer
        buffer[:, write_idx] = np.where(spikes, SPIKE_VOLTAGE, REST_VOLTAGE)

        # Read delayed outputs (delay_steps behind)
        read_idx = (t + 1) % DELAY_STEPS
        outputs = buffer[:, read_idx]

        # Count output spikes (prevents dead code elimination)
        total_output_spikes += np.sum(outputs > 0)

    elapsed = time.perf_counter() - start

    return {
        "method": "dense_ring_buffer",
        "n_neurons": n_neurons,
        "buffer_bytes": buffer_bytes,
        "buffer_mb": buffer_bytes / (1024 * 1024),
        "elapsed_s": elapsed,
        "steps_per_sec": SIMULATION_STEPS / elapsed,
        "total_output_spikes": int(total_output_spikes),
    }


# =============================================================================
# Method B: Sparse Event Queue
# =============================================================================


def _benchmark_sparse_event_queue(n_neurons: int) -> dict:
    """
    Sparse event-based approach.

    Memory: O(active_spikes) - only stores spike events, not full voltage.
    Uses a list of (delivery_time, neuron_id) tuples.
    """
    from collections import deque

    # Event queue: deque of (delivery_step, neuron_indices)
    # We'll use a dict mapping delivery_step -> set of neuron IDs
    event_queue: dict[int, set[int]] = {}

    # Track peak memory (approximate)
    peak_queue_size = 0

    # Output accumulator
    total_output_spikes = 0

    start = time.perf_counter()

    for t in range(SIMULATION_STEPS):
        # Generate random spikes
        spikes = np.random.random(n_neurons) < FIRING_PROB
        spike_indices = np.where(spikes)[0]

        # Schedule delivery
        delivery_time = t + DELAY_STEPS
        if len(spike_indices) > 0:
            if delivery_time not in event_queue:
                event_queue[delivery_time] = set()
            event_queue[delivery_time].update(spike_indices)

        # Process deliveries for current time
        if t in event_queue:
            total_output_spikes += len(event_queue[t])
            del event_queue[t]

        # Track peak queue size
        current_size = sum(len(v) for v in event_queue.values())
        peak_queue_size = max(peak_queue_size, current_size)

    elapsed = time.perf_counter() - start

    # Estimate memory: each event is roughly (int + set overhead)
    # Very rough estimate: ~100 bytes per active spike in queue
    estimated_bytes = peak_queue_size * 100

    return {
        "method": "sparse_event_queue",
        "n_neurons": n_neurons,
        "buffer_bytes": estimated_bytes,
        "buffer_mb": estimated_bytes / (1024 * 1024),
        "elapsed_s": elapsed,
        "steps_per_sec": SIMULATION_STEPS / elapsed,
        "total_output_spikes": int(total_output_spikes),
        "peak_queue_size": peak_queue_size,
    }


# =============================================================================
# Method C: Numpy Sparse (CSR-like approach)
# =============================================================================


def _benchmark_numpy_sparse(n_neurons: int) -> dict:
    """
    NumPy-based sparse approach using boolean arrays.

    Memory: O(N_neurons × delay_steps) but as bool (1 byte vs 8 bytes)
    More cache-friendly than event queue for dense-ish firing.
    """
    # Boolean buffer: (N_neurons, delay_steps) - 1 byte per entry
    buffer = np.zeros((n_neurons, DELAY_STEPS), dtype=np.bool_)

    buffer_bytes = buffer.nbytes
    total_output_spikes = 0

    start = time.perf_counter()

    for t in range(SIMULATION_STEPS):
        # Generate random spikes
        spikes = np.random.random(n_neurons) < FIRING_PROB

        write_idx = t % DELAY_STEPS
        read_idx = (t + 1) % DELAY_STEPS

        # Write spikes as boolean
        buffer[:, write_idx] = spikes

        # Read delayed outputs
        outputs = buffer[:, read_idx]
        total_output_spikes += np.sum(outputs)

    elapsed = time.perf_counter() - start

    return {
        "method": "numpy_sparse_bool",
        "n_neurons": n_neurons,
        "buffer_bytes": buffer_bytes,
        "buffer_mb": buffer_bytes / (1024 * 1024),
        "elapsed_s": elapsed,
        "steps_per_sec": SIMULATION_STEPS / elapsed,
        "total_output_spikes": int(total_output_spikes),
    }


# =============================================================================
# Method D: Current fast_model Implementation
# =============================================================================


def _benchmark_current_fast_model(n_neurons: int) -> dict:
    """
    Benchmark the current fast_model implementation.

    This uses the actual simulate_fast_model function which stores
    the entire waveform for each neuron.

    Memory: O(N_neurons × waveform_length) float64
    """
    # Generate time array for full simulation
    T_ms = SIMULATION_STEPS * DT_MS  # 100ms
    t_ms = np.arange(0, T_ms, DT_MS)
    waveform_length = len(t_ms)

    # Generate random spike train for each neuron
    # Each neuron gets a random waveform with spikes at ~FIRING_PROB rate
    # We'll create one representative waveform and process N neurons

    # Memory for all input waveforms: N × waveform_length × 8 bytes
    # This is the dominant memory cost
    input_bytes = n_neurons * waveform_length * 8  # float64

    total_output_spikes = 0
    total_blocked = 0

    start = time.perf_counter()

    # Process neurons in batches to avoid massive memory allocation
    BATCH_SIZE = min(1000, n_neurons)
    n_batches = (n_neurons + BATCH_SIZE - 1) // BATCH_SIZE

    for batch in range(n_batches):
        batch_start = batch * BATCH_SIZE
        batch_end = min((batch + 1) * BATCH_SIZE, n_neurons)
        batch_n = batch_end - batch_start

        # Generate input waveforms for this batch
        v_inputs = np.full((batch_n, waveform_length), REST_VOLTAGE, dtype=np.float64)

        # Add random spikes
        spike_mask = np.random.random((batch_n, waveform_length)) < FIRING_PROB
        v_inputs[spike_mask] = SPIKE_VOLTAGE

        # Process each neuron through fast_model
        for i in range(batch_n):
            result = simulate_fast_model(
                v_input=v_inputs[i],
                t_ms_input=t_ms,
                delay_ms=DELAY_MS,
                v_rest=REST_VOLTAGE,
                refractory_period_ms=5.0,
                spike_threshold_mv=-20.0,
            )
            # Count output spikes
            total_output_spikes += np.sum(result["V"] > 0)
            total_blocked += result["blocked_count"]

    elapsed = time.perf_counter() - start

    return {
        "method": "current_fast_model",
        "n_neurons": n_neurons,
        "buffer_bytes": input_bytes,
        "buffer_mb": input_bytes / (1024 * 1024),
        "elapsed_s": elapsed,
        "steps_per_sec": (n_neurons * SIMULATION_STEPS) / elapsed,
        "total_output_spikes": int(total_output_spikes),
        "total_blocked": int(total_blocked),
        "waveform_length": waveform_length,
    }


def _benchmark_current_fast_model_single_call(n_neurons: int) -> dict:
    """
    Benchmark fast_model with a single representative waveform called N times.

    This measures the per-call overhead without the memory cost of
    storing N different waveforms.
    """
    # Generate time array
    T_ms = SIMULATION_STEPS * DT_MS
    t_ms = np.arange(0, T_ms, DT_MS)
    waveform_length = len(t_ms)

    # Generate one representative waveform with random spikes
    v_input = np.full(waveform_length, REST_VOLTAGE, dtype=np.float64)
    spike_mask = np.random.random(waveform_length) < FIRING_PROB
    v_input[spike_mask] = SPIKE_VOLTAGE

    # Memory: just one waveform + output
    buffer_bytes = waveform_length * 8 * 2  # input + output

    total_output_spikes = 0
    total_blocked = 0

    start = time.perf_counter()

    for _ in range(n_neurons):
        result = simulate_fast_model(
            v_input=v_input,
            t_ms_input=t_ms,
            delay_ms=DELAY_MS,
            v_rest=REST_VOLTAGE,
            refractory_period_ms=5.0,
            spike_threshold_mv=-20.0,
        )
        total_output_spikes += np.sum(result["V"] > 0)
        total_blocked += result["blocked_count"]

    elapsed = time.perf_counter() - start

    return {
        "method": "fast_model_single_waveform",
        "n_neurons": n_neurons,
        "buffer_bytes": buffer_bytes,
        "buffer_mb": buffer_bytes / (1024 * 1024),
        "elapsed_s": elapsed,
        "neurons_per_sec": n_neurons / elapsed,
        "total_output_spikes": int(total_output_spikes),
        "total_blocked": int(total_blocked),
    }


# =============================================================================
# Main Benchmark Runner
# =============================================================================


def _run_benchmarks() -> list[dict]:
    """Run all benchmarks and collect results."""
    results = []

    print(f"Memory Benchmark Configuration:")
    print(f"  delay_steps: {DELAY_STEPS} (5ms at 0.01ms dt)")
    print(f"  simulation_steps: {SIMULATION_STEPS} (100ms)")
    print(f"  firing_prob: {FIRING_PROB} (~{FIRING_PROB * 100000:.0f} Hz)")
    print(f"  timeout: {TIMEOUT_SECONDS}s")
    print()

    for n in N_NEURON_COUNTS:
        print(f"\n{'=' * 70}")
        print(f"N = {n:,} neurons")
        print(f"{'=' * 70}")

        # Method A: Dense Ring Buffer
        print(f"\n  [A] Dense Ring Buffer (float64)...")
        try:
            result_a = _benchmark_dense_ring_buffer(n)
            results.append(result_a)
            print(f"      Memory: {result_a['buffer_mb']:.1f} MB")
            print(
                f"      Time: {result_a['elapsed_s']:.3f}s ({result_a['steps_per_sec']:.0f} steps/s)"
            )
        except MemoryError:
            print(f"      FAILED: Out of memory")
            results.append(
                {"method": "dense_ring_buffer", "n_neurons": n, "error": "OOM"}
            )

        # Method B: Sparse Event Queue
        print(f"\n  [B] Sparse Event Queue...")
        try:
            result_b = _benchmark_sparse_event_queue(n)
            results.append(result_b)
            print(f"      Memory (est.): {result_b['buffer_mb']:.1f} MB")
            print(f"      Peak queue: {result_b['peak_queue_size']:,} events")
            print(
                f"      Time: {result_b['elapsed_s']:.3f}s ({result_b['steps_per_sec']:.0f} steps/s)"
            )
        except MemoryError:
            print(f"      FAILED: Out of memory")
            results.append(
                {"method": "sparse_event_queue", "n_neurons": n, "error": "OOM"}
            )

        # Method C: Boolean Buffer
        print(f"\n  [C] NumPy Boolean Buffer...")
        try:
            result_c = _benchmark_numpy_sparse(n)
            results.append(result_c)
            print(f"      Memory: {result_c['buffer_mb']:.1f} MB")
            print(
                f"      Time: {result_c['elapsed_s']:.3f}s ({result_c['steps_per_sec']:.0f} steps/s)"
            )
        except MemoryError:
            print(f"      FAILED: Out of memory")
            results.append(
                {"method": "numpy_sparse_bool", "n_neurons": n, "error": "OOM"}
            )

        # Method D: Current fast_model (single waveform, N calls)
        # Only run for smaller N to avoid excessive time
        if n <= 10_000:
            print(f"\n  [D] Current fast_model (N calls, same waveform)...")
            try:
                result_d = _benchmark_current_fast_model_single_call(n)
                results.append(result_d)
                print(f"      Memory (per call): {result_d['buffer_mb']:.1f} MB")
                print(f"      Blocked spikes: {result_d['total_blocked']:,}")
                print(
                    f"      Time: {result_d['elapsed_s']:.3f}s ({result_d['neurons_per_sec']:.0f} neurons/s)"
                )
            except MemoryError:
                print(f"      FAILED: Out of memory")
                results.append(
                    {
                        "method": "fast_model_single_waveform",
                        "n_neurons": n,
                        "error": "OOM",
                    }
                )
        else:
            print(f"\n  [D] Current fast_model - SKIPPED (N > 10K, too slow)")
            results.append(
                {
                    "method": "fast_model_single_waveform",
                    "n_neurons": n,
                    "error": "SKIPPED",
                }
            )

    return results


def _print_summary(results: list[dict]):
    """Print summary table."""
    print("\n")
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)

    # Group by method
    methods = [
        "dense_ring_buffer",
        "sparse_event_queue",
        "numpy_sparse_bool",
        "fast_model_single_waveform",
    ]
    method_labels = {
        "dense_ring_buffer": "Dense (f64)",
        "sparse_event_queue": "Sparse Queue",
        "numpy_sparse_bool": "Bool Buffer",
        "fast_model_single_waveform": "fast_model",
    }

    print(f"\n{'N neurons':>12} | ", end="")
    for m in methods:
        print(f"{method_labels[m]:>20} | ", end="")
    print()
    print("-" * 110)

    for n in N_NEURON_COUNTS:
        print(f"{n:>12,} | ", end="")
        for m in methods:
            r = next(
                (
                    x
                    for x in results
                    if x.get("method") == m and x.get("n_neurons") == n
                ),
                None,
            )
            if r and "error" not in r:
                print(f"{r['buffer_mb']:>8.1f} MB {r['elapsed_s']:>6.2f}s | ", end="")
            elif r and r.get("error") == "OOM":
                print(f"{'OOM':>20} | ", end="")
            elif r and r.get("error") == "SKIPPED":
                print(f"{'SKIPPED':>20} | ", end="")
            else:
                print(f"{'N/A':>20} | ", end="")
        print()

    print("=" * 110)

    # Memory scaling analysis
    print("\nMemory Scaling (theoretical):")
    print(
        f"  Dense (float64):  N × {DELAY_STEPS} × 8 bytes = N × {DELAY_STEPS * 8 / 1024:.1f} KB"
    )
    print(
        f"  Bool buffer:      N × {DELAY_STEPS} × 1 byte  = N × {DELAY_STEPS / 1024:.1f} KB"
    )
    print(f"  Sparse queue:     ~N × firing_rate × delay   = variable")
    print(
        f"  fast_model:       N × {SIMULATION_STEPS} × 8 bytes = N × {SIMULATION_STEPS * 8 / 1024:.1f} KB (full waveform!)"
    )

    print("\nKey Insight:")
    print(
        f"  Current fast_model stores FULL waveform ({SIMULATION_STEPS} points) per neuron"
    )
    print(f"  Ring buffer only stores delay window ({DELAY_STEPS} points) per neuron")
    print(
        f"  Memory ratio: {SIMULATION_STEPS / DELAY_STEPS:.0f}x more for fast_model vs ring buffer"
    )

    print("\nRecommendation:")
    print("  - For batch processing (current use): fast_model is fine")
    print("  - For real-time streaming: Use ring buffer or event queue")
    print("  - For large-scale (>10K neurons): Consider sparse approaches")


def main():
    print("=" * 70)
    print("MEMORY BENCHMARK: Fast Model Delay Buffer Strategies")
    print("=" * 70)

    results = _run_benchmarks()
    _print_summary(results)


if __name__ == "__main__":
    main()
