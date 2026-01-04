"""
Neuron Scaling Benchmark

Measures simulation time as we scale the number of neurons (N).
Demonstrates that Fast Model is orders of magnitude faster than HH Cable.

Models tested:
- hh_cable: Full Hodgkin-Huxley cable equation (physics baseline)
- fast_model: Pure delay-line model (proposed solution)
"""

import time
import numpy as np

from hh_cable import simulate_hh_cable
from fast_model import simulate_fast_model

# =============================================================================
# Configuration
# =============================================================================

T_MS = 100.0  # Simulation duration (ms)
T_S = T_MS * 1e-3  # Simulation duration (s)

# HH Cable optimized parameters (from benchmark_shapes.py)
HH_CABLE_PARAMS = {
    "L": 5000.0,  # μm
    "dx": 50.0,  # μm
    "dt_s": 1e-5,  # s (0.01 ms)
    "stim_duration_s": 5e-3,  # 5 ms
    "stim_amplitude": 1000.0,
    "stim_index": 1,
    "store_history": False,  # Don't store history for speed
    "history_stride": 1,
}

# Fast model parameters
FAST_MODEL_DELAY_MS = 10.0  # Fixed delay

# Scaling test parameters
N_COUNTS = [1, 10, 50, 100, 500, 1000, 5000, 10000]
TIMEOUT_SECONDS = 5.0  # Skip larger N if previous took longer than this


# =============================================================================
# Pre-generate input waveform for fast_model
# =============================================================================


def _generate_sample_waveform(
    n_points: int, dt_ms: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a sample AP waveform for fast_model input.
    Uses a simple parametric shape (not full HH simulation for speed).
    """
    t_ms = np.arange(n_points) * dt_ms

    # Simple AP-like waveform: fast rise, slower fall
    # Parameters tuned to match typical HH AP shape
    peak_time = 10.0  # ms
    rise_tau = 0.3  # ms
    fall_tau = 1.5  # ms

    v_rest = -65.0
    v_peak = 40.0
    v_undershoot = -75.0

    # Rising phase (sigmoid)
    rise = (v_peak - v_rest) / (1 + np.exp(-(t_ms - peak_time) / rise_tau))
    # Falling phase (exponential decay after peak)
    fall_mask = t_ms > peak_time
    fall = np.zeros_like(t_ms)
    fall[fall_mask] = (v_peak - v_undershoot) * np.exp(
        -(t_ms[fall_mask] - peak_time) / fall_tau
    )

    # Combine: rise up to peak, then decay to undershoot
    V = np.where(t_ms <= peak_time, v_rest + rise, v_undershoot + fall)

    return t_ms, V


# =============================================================================
# Benchmark functions
# =============================================================================


def _time_hh_cable(n_neurons: int) -> float:
    """Time running hh_cable simulation for N neurons sequentially."""
    start = time.perf_counter()
    for _ in range(n_neurons):
        simulate_hh_cable(
            T_s=T_S,
            **HH_CABLE_PARAMS,
        )
    elapsed = time.perf_counter() - start
    return elapsed


def _time_fast_model(
    n_neurons: int, t_ms_input: np.ndarray, v_input: np.ndarray
) -> float:
    """Time running fast_model simulation for N neurons sequentially."""
    start = time.perf_counter()
    for _ in range(n_neurons):
        simulate_fast_model(
            v_input=v_input,
            t_ms_input=t_ms_input,
            delay_ms=FAST_MODEL_DELAY_MS,
        )
    elapsed = time.perf_counter() - start
    return elapsed


def _run_scaling_benchmark() -> dict:
    """
    Run the scaling benchmark for all models.

    Returns dict mapping model_name -> list of (n, time) tuples
    """
    # Pre-generate input waveform for fast_model
    # Use same time resolution as HH cable
    dt_ms = HH_CABLE_PARAMS["dt_s"] * 1e3
    n_points = int(T_MS / dt_ms)
    t_ms_input, v_input = _generate_sample_waveform(n_points, dt_ms)

    results = {
        "hh_cable": [],
        "fast_model": [],
    }

    # Track if we should skip due to timeout
    skip_hh_cable = False
    skip_fast_model = False

    print(f"\nScaling Benchmark: T = {T_MS} ms, timeout = {TIMEOUT_SECONDS}s")
    print("=" * 70)
    print(f"{'N':>8}  {'HH Cable (s)':>14}  {'Fast Model (s)':>14}  {'Speedup':>10}")
    print("-" * 70)

    for n in N_COUNTS:
        hh_time = None
        fast_time = None

        # HH Cable
        if not skip_hh_cable:
            print(f"{n:>8}  ", end="", flush=True)
            hh_time = _time_hh_cable(n)
            results["hh_cable"].append((n, hh_time))
            print(f"{hh_time:>14.4f}  ", end="", flush=True)

            if hh_time > TIMEOUT_SECONDS:
                skip_hh_cable = True
                print("(timeout) ", end="")
        else:
            print(f"{n:>8}  {'skipped':>14}  ", end="", flush=True)

        # Fast Model
        if not skip_fast_model:
            fast_time = _time_fast_model(n, t_ms_input, v_input)
            results["fast_model"].append((n, fast_time))
            print(f"{fast_time:>14.6f}  ", end="", flush=True)

            if fast_time > TIMEOUT_SECONDS:
                skip_fast_model = True
        else:
            print(f"{'skipped':>14}  ", end="", flush=True)

        # Speedup factor
        if hh_time is not None and fast_time is not None:
            speedup = hh_time / fast_time
            print(f"{speedup:>10.1f}x")
        else:
            print(f"{'---':>10}")

    print("=" * 70)

    return results


# =============================================================================
# Visualization
# =============================================================================


def _plot_scaling(results: dict, output_path: str):
    """Create log-log scaling plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot styling
    styles = {
        "hh_cable": {
            "color": "black",
            "marker": "o",
            "markersize": 8,
            "linewidth": 2,
            "label": "HH Cable (physics baseline)",
        },
        "fast_model": {
            "color": "red",
            "marker": "s",
            "markersize": 8,
            "linewidth": 2,
            "label": "Fast Model (delay-line)",
        },
    }

    for model_name, data in results.items():
        if not data:
            continue
        n_values = [d[0] for d in data]
        times = [d[1] for d in data]
        style = styles.get(model_name, {})
        ax.loglog(n_values, times, **style)

    # Add reference lines
    n_ref = np.array([1, 10000])
    # O(N) reference
    ax.loglog(
        n_ref, n_ref * 1e-5, "g--", alpha=0.5, linewidth=1, label="O(N) reference"
    )

    ax.set_xlabel("Number of Neurons (N)", fontsize=12)
    ax.set_ylabel("Simulation Time (seconds)", fontsize=12)
    ax.set_title(f"Neuron Scaling Benchmark (T = {T_MS} ms)", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    # Add timeout line
    ax.axhline(
        TIMEOUT_SECONDS,
        color="orange",
        linestyle=":",
        alpha=0.7,
        label=f"Timeout ({TIMEOUT_SECONDS}s)",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)


def _print_summary(results: dict):
    """Print speedup summary for largest shared N."""
    hh_data = results.get("hh_cable", [])
    fast_data = results.get("fast_model", [])

    if not hh_data or not fast_data:
        print("\nInsufficient data for speedup summary.")
        return

    # Find largest shared N
    hh_ns = {d[0]: d[1] for d in hh_data}
    fast_ns = {d[0]: d[1] for d in fast_data}

    shared_ns = sorted(set(hh_ns.keys()) & set(fast_ns.keys()))

    if not shared_ns:
        print("\nNo shared N values for speedup comparison.")
        return

    print("\n" + "=" * 50)
    print("SPEEDUP SUMMARY")
    print("=" * 50)

    for n in shared_ns:
        hh_time = hh_ns[n]
        fast_time = fast_ns[n]
        speedup = hh_time / fast_time
        print(f"N = {n:>5}: HH Cable = {hh_time:.4f}s, Fast Model = {fast_time:.6f}s")
        print(f"          Speedup = {speedup:.1f}x")

    # Highlight largest N
    largest_n = shared_ns[-1]
    final_speedup = hh_ns[largest_n] / fast_ns[largest_n]
    print("-" * 50)
    print(
        f"At N = {largest_n}: Fast Model is {final_speedup:.0f}x faster than HH Cable"
    )
    print("=" * 50)

    # Project time for larger N if we have data
    if len(fast_data) > len(hh_data):
        largest_fast_n = fast_data[-1][0]
        largest_fast_time = fast_data[-1][1]

        # Estimate HH time by extrapolation (assuming linear scaling)
        if len(hh_data) >= 2:
            # Use last two HH points to estimate slope
            n1, t1 = hh_data[-2]
            n2, t2 = hh_data[-1]
            time_per_neuron = (t2 - t1) / (n2 - n1)
            estimated_hh_time = t2 + time_per_neuron * (largest_fast_n - n2)

            print(f"\nProjection for N = {largest_fast_n}:")
            print(f"  Fast Model actual:    {largest_fast_time:.4f}s")
            print(
                f"  HH Cable estimated:   {estimated_hh_time:.1f}s ({estimated_hh_time / 60:.1f} min)"
            )
            print(
                f"  Estimated speedup:    {estimated_hh_time / largest_fast_time:.0f}x"
            )


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("NEURON SCALING BENCHMARK")
    print("=" * 70)
    print(f"Testing: hh_cable vs fast_model")
    print(f"Duration: {T_MS} ms per simulation")
    print(f"N values: {N_COUNTS}")
    print(f"Timeout: {TIMEOUT_SECONDS}s (skip larger N if exceeded)")

    # Run benchmark
    results = _run_scaling_benchmark()

    # Plot results
    _plot_scaling(results, "speed_scaling.png")

    # Print summary
    _print_summary(results)


if __name__ == "__main__":
    main()
