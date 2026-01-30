"""
Axon Conduction Speed Benchmark - Bar Chart Comparison

Compares wall-clock time to propagate signals through axonal models:
1. Passive Cable (100-compartment passive cable equation)
2. Event Model (PCA-based encode/decode with delay-line)

Both models use the same pre-generated input signal (50Hz spike train).
This isolates the axon conduction cost without spike generation overhead.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import time
import numpy as np


# =============================================================================
# Generate Common Input Signal (50Hz spike train, 1000ms)
# =============================================================================


def generate_50hz_train(
    duration_ms: float = 1000.0, dt_ms: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 50Hz spike train using HH dynamics.

    Returns (t_ms, V) arrays.
    """
    from math import exp as math_exp

    # HH parameters
    C_m = 1.0
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    v_rest = -65.0

    stim_amplitude = 30.0
    stim_duration_ms = 1.0

    # 50Hz = 20ms ISI
    freq_hz = 50.0
    isi_ms = 1000.0 / freq_hz
    n_pulses = int(duration_ms / isi_ms)

    # Pulse times starting at 10ms
    pulse_times = [10.0 + i * isi_ms for i in range(n_pulses)]

    n_time = int(duration_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms

    # Create stimulus array
    i_stim = np.zeros(n_time)
    for t_pulse in pulse_times:
        start_idx = int(t_pulse / dt_ms)
        end_idx = int((t_pulse + stim_duration_ms) / dt_ms)
        end_idx = min(end_idx, n_time)
        if start_idx < n_time:
            i_stim[start_idx:end_idx] = stim_amplitude

    # Initialize state
    V_val = v_rest
    m_val = 0.0529
    h_val = 0.5961
    n_val = 0.3177

    inv_C_m = 1.0 / C_m
    V_hist = np.zeros(n_time)
    V_hist[0] = V_val

    # Integration loop
    for i in range(1, n_time):
        I_Na = g_Na * (m_val**3) * h_val * (V_val - E_Na)
        I_K = g_K * (n_val**4) * (V_val - E_K)
        I_L = g_L * (V_val - E_L)

        V_p55 = V_val + 55.0
        V_p40 = V_val + 40.0
        V_p65 = V_val + 65.0
        V_p35 = V_val + 35.0

        a_n = -0.01 * V_p55 / (math_exp(V_p55 / -10.0) - 1.0)
        b_n = 0.125 * math_exp(V_p65 / -80.0)
        a_m = -0.1 * V_p40 / (math_exp(V_p40 / -10.0) - 1.0)
        b_m = 4.0 * math_exp(V_p65 / -18.0)
        a_h = 0.07 * math_exp(V_p65 / -20.0)
        b_h = 1.0 / (1.0 + math_exp(V_p35 / -10.0))

        dm = a_m * (1.0 - m_val) - b_m * m_val
        dh = a_h * (1.0 - h_val) - b_h * h_val
        dn = a_n * (1.0 - n_val) - b_n * n_val

        m_val += dm * dt_ms
        h_val += dh * dt_ms
        n_val += dn * dt_ms

        I_stim_val = i_stim[i - 1]
        dV = (I_stim_val - I_Na - I_K - I_L) * inv_C_m
        V_val += dV * dt_ms

        V_hist[i] = V_val

    return t_ms, V_hist


# =============================================================================
# Benchmark Functions
# =============================================================================


def benchmark_passive_cable(
    v_source: np.ndarray, t_ms: np.ndarray, n_iterations: int = 10
) -> float:
    """
    Benchmark Passive Cable model with v_source as boundary condition.

    Config: L=5000μm, dx=50μm (100 compartments), dt=0.01ms
    """
    from src.cable_equation import simulate_cable_equation

    T_ms = t_ms[-1]
    dt_ms = t_ms[1] - t_ms[0]
    dx = 50.0
    L = 5000.0

    times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()

        simulate_cable_equation(
            L=L,
            T_ms=T_ms,
            dx=dx,
            dt_ms=dt_ms,
            v_input=v_source,
            t_ms_input=t_ms,
            store_history=False,
        )

        t_end = time.perf_counter()
        times.append(t_end - t_start)

    return float(np.mean(times))


def benchmark_event_model(
    v_source: np.ndarray, t_ms: np.ndarray, n_iterations: int = 10
) -> float:
    """
    Benchmark Event (PCA-based) model.

    Includes: spike detection → encoding → queueing → decoding.
    """
    from src.event_model import EventPropagator

    # Same delay as passive cable would produce
    delay_ms = 50.0
    prop = EventPropagator(delay_ms=delay_ms)

    times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()

        prop.simulate(v_source, t_ms)

        t_end = time.perf_counter()
        times.append(t_end - t_start)

    return float(np.mean(times))


# =============================================================================
# Main
# =============================================================================


def main():
    import matplotlib.pyplot as plt

    # Use Inter font for all text
    plt.rcParams["font.family"] = "Inter"

    print("=" * 70)
    print("Axon Conduction Speed Benchmark")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Duration: 1000ms")
    print("  Input: 50Hz spike train (~50 spikes)")
    print("  Axon length: 5000μm")
    print("  Passive Cable: 100 compartments (dx=50μm)")
    print("  Time step: 0.01ms")
    print("\nThis benchmark compares AXON CONDUCTION only (no spike generation)")

    # Generate common input signal
    print("\n[1/3] Generating 50Hz spike train...")
    t_ms, v_source = generate_50hz_train(duration_ms=1000.0, dt_ms=0.01)
    n_spikes = np.sum(np.diff((v_source > -20).astype(int)) > 0)
    print(f"       Generated {len(v_source)} samples, {n_spikes} spikes detected")

    n_iter = 10
    results = {}

    # Benchmark Passive Cable
    print(f"\n[2/3] Benchmarking Passive Cable ({n_iter} iterations)...")
    try:
        results["Passive Cable"] = benchmark_passive_cable(
            v_source, t_ms, n_iterations=n_iter
        )
        print(f"       Average time: {results['Passive Cable']:.4f}s")
    except Exception as e:
        print(f"       ERROR: {e}")
        results["Passive Cable"] = None

    # Benchmark Event Model
    print(f"\n[3/3] Benchmarking Event Model ({n_iter} iterations)...")
    try:
        results["Event (PCA)"] = benchmark_event_model(
            v_source, t_ms, n_iterations=n_iter
        )
        print(f"       Average time: {results['Event (PCA)']:.6f}s")
    except Exception as e:
        print(f"       ERROR: {e}")
        results["Event (PCA)"] = None

    # Filter out failed benchmarks
    valid_results = {k: v for k, v in results.items() if v is not None}

    if not valid_results:
        print("\nERROR: No benchmarks succeeded!")
        return

    # Calculate speedups relative to Passive Cable (baseline)
    baseline_name = "Passive Cable" if "Passive Cable" in valid_results else None
    if baseline_name is None:
        baseline_name = max(valid_results.keys(), key=lambda k: valid_results[k])
    baseline_time = valid_results[baseline_name]

    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"{'Model':<20} {'Time (s)':<15} {'Speedup':<15}")
    print("-" * 70)

    speedups = {}
    for name, t in valid_results.items():
        speedup = baseline_time / t
        speedups[name] = speedup
        speedup_str = (
            f"{speedup:.0f}x" if speedup >= 1 else f"{1 / speedup:.1f}x slower"
        )
        print(f"{name:<20} {t:<15.6f} {speedup_str:<15}")

    print("-" * 70)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(6, 5))

    models = list(valid_results.keys())
    times = [valid_results[m] for m in models]

    # Colors (Seaborn deep palette)
    color_map = {
        "Passive Cable": "#55a868",  # Deep green
        "Event (PCA)": "#4c72b0",  # Deep blue
    }
    colors = [color_map.get(m, "#8c8c8c") for m in models]

    # Create bars
    bars = ax.bar(models, times, color=colors, edgecolor="black", linewidth=1.2)

    # Log scale
    ax.set_yscale("log")

    # Add speedup annotations above bars
    for bar, model in zip(bars, models):
        height = bar.get_height()
        speedup = speedups[model]

        if speedup >= 1000:
            label = f"{speedup / 1000:.1f}k×"
        elif speedup >= 1:
            label = f"{speedup:.0f}×"
        else:
            label = f"{speedup:.2f}×"

        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=color_map.get(model, "#1a1a1a"),
        )

    # Labels and title
    ax.set_ylabel("Time to simulate 1s (seconds)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Axon Conduction Speed Benchmark\n"
        f"(5000μm axon, 50Hz input, {n_iter} iterations average)",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.yaxis.grid(True, alpha=0.3, which="both")
    ax.set_axisbelow(True)

    # Add horizontal reference lines
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(y=0.001, color="gray", linestyle=":", alpha=0.3, linewidth=1)

    # Adjust y-axis limits to give room for annotations
    y_max = max(times) * 3
    y_min = min(times) / 3
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    output_file = _OUTPUT_DIR / "speed_benchmark_bar.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {output_file}")
    plt.close()

    # Summary
    if "Passive Cable" in valid_results and "Event (PCA)" in valid_results:
        event_speedup = valid_results["Passive Cable"] / valid_results["Event (PCA)"]
        print(f"\n{'=' * 70}")
        print(
            f"KEY RESULT: Event Model is {event_speedup:.0f}× faster than Passive Cable"
        )
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
