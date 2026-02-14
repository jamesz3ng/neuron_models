"""
Speedup Comparison Benchmark: Event-Based vs Traditional Models

Bar graph comparing how many times slower traditional PDE-based models are
compared to the Event-Based model at two spatial resolutions (N=50 and N=10,000).

Models compared:
- HH: HH Cable (Nonlinear PDE)
- PC: Passive Cable (Linear PDE)
- WV: Hybrid Wave (Wave Equation)

Baseline: Event-Based model (O(1) complexity)
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

from src.hh_cable import simulate_hh_cable
from src.cable_equation import simulate_cable_equation
from src.wave_model import simulate_wave_model
from src.event_model import EventPropagator
from src.ais_simulation import run_2comp_simulation
from src.simulation import create_pulse_train

# =============================================================================
# CONFIGURATION
# =============================================================================

T_DURATION_MS = 100.0  # 100ms simulation
DT_MS = 0.001  # Fixed 1us time step for stability
L_UM = 5000.0  # 5mm axon

# Spatial nodes to test (only two samples)
N_VALUES = [50, 10000]

# Number of iterations for averaging
N_ITER = 5

# Model abbreviations for bar labels
MODEL_ABBREV = {
    "HH Cable": "HH",
    "Passive Cable": "PC",
    "Hybrid Wave": "WV",
}

# Full names for legend
MODEL_FULL_NAMES = {
    "HH Cable": "HH Cable (Nonlinear PDE)",
    "Passive Cable": "Passive Cable (Linear PDE)",
    "Hybrid Wave": "Hybrid Wave (Wave Equation)",
}

# Colors for each model - Soft pastel palette (inspired by #ffe6cc, #fff2cc, #dae8fc)
COLORS = {
    "HH Cable": "#ffb380",  # Warm peach/orange
    "Passive Cable": "#ffd966",  # Soft golden yellow
    "Hybrid Wave": "#9fc5e8",  # Soft sky blue
}

# =============================================================================
# PRE-CALCULATE TRIGGER SIGNAL
# =============================================================================


def prepare_input_signal():
    """Generate realistic AIS-driven input waveform."""
    print("Pre-calculating source trigger signal (100ms)...")
    pulse_times = [10.0, 30.0, 50.0, 70.0, 90.0]  # 5 spikes
    i_stim = create_pulse_train(T_DURATION_MS, pulse_times, dt_ms=0.002)
    t_ms_src, vs_src, va_src, *_ = run_2comp_simulation(
        T_DURATION_MS, i_stim, dt_ms=0.002
    )

    # Resample to 0.001ms dt for the benchmark models
    t_ms_1us = np.arange(0, T_DURATION_MS, DT_MS)
    va_src_1us = np.interp(t_ms_1us, t_ms_src, va_src)

    return t_ms_1us, va_src_1us


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def benchmark(t_ms: np.ndarray, va_src: np.ndarray) -> dict:
    """Run benchmarks and return speedup factors for each model at each N."""

    results = {n: {} for n in N_VALUES}

    # Init Propagator once (O(1) setup)
    prop = EventPropagator(delay_ms=0.0)

    for n in N_VALUES:
        print(f"\nBenchmarking N = {n:,} nodes...")
        dx = L_UM / n

        # 1. Event-Based (baseline)
        print("  - Event-Based (baseline)...")
        iters = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            prop.simulate(va_src, t_ms)
            iters.append(time.perf_counter() - t0)
        event_time = np.mean(iters)
        results[n]["Event-Based"] = event_time
        print(f"    Time: {event_time:.6f}s")

        # 2. HH Cable
        print("  - HH Cable...")
        iters = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            simulate_hh_cable(
                L=L_UM,
                dx=dx,
                dt_s=DT_MS / 1000.0,
                T_s=T_DURATION_MS / 1000.0,
                stim_waveform=va_src,
                t_s_stim=t_ms / 1000.0,
                stim_index=0,
                store_history=False,
            )
            iters.append(time.perf_counter() - t0)
        hh_time = np.mean(iters)
        results[n]["HH Cable"] = hh_time / event_time
        print(f"    Time: {hh_time:.6f}s  ->  {hh_time / event_time:.1f}x slower")

        # 3. Passive Cable
        print("  - Passive Cable...")
        iters = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            simulate_cable_equation(
                L=L_UM,
                dx=dx,
                dt_ms=DT_MS,
                T_ms=T_DURATION_MS,
                v_input=va_src,
                t_ms_input=t_ms,
                store_history=False,
            )
            iters.append(time.perf_counter() - t0)
        pc_time = np.mean(iters)
        results[n]["Passive Cable"] = pc_time / event_time
        print(f"    Time: {pc_time:.6f}s  ->  {pc_time / event_time:.1f}x slower")

        # 4. Hybrid Wave
        print("  - Hybrid Wave...")
        iters = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            simulate_wave_model(
                L=L_UM,
                nx=n,
                dt_s=DT_MS / 1000.0,
                T_s=T_DURATION_MS / 1000.0,
                v_input=va_src,
                t_input=t_ms / 1000.0,
                store_history=False,
            )
            iters.append(time.perf_counter() - t0)
        wv_time = np.mean(iters)
        results[n]["Hybrid Wave"] = wv_time / event_time
        print(f"    Time: {wv_time:.6f}s  ->  {wv_time / event_time:.1f}x slower")

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_speedup_comparison(results: dict):
    """Create grouped bar chart comparing speedup factors."""

    fig, ax = plt.subplots(figsize=(12, 8))

    # Bar configuration - now includes Event-Based as a bar
    model_names = ["Event-Based", "HH Cable", "Passive Cable", "Hybrid Wave"]
    n_models = len(model_names)
    bar_width = 0.2

    # X positions for each group
    x_positions = np.arange(len(N_VALUES))

    # Create bars for each model
    for i, model in enumerate(model_names):
        if model == "Event-Based":
            speedups = [1.0 for _ in N_VALUES]  # Baseline is always 1x
            color = "#93c47d"  # Soft sage green
            label = "EB = Event-Based (Baseline)"
        else:
            speedups = [results[n][model] for n in N_VALUES]
            color = COLORS[model]
            label = f"{MODEL_ABBREV[model]} = {MODEL_FULL_NAMES[model]}"

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x_positions + offset,
            speedups,
            bar_width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on top of bars
        for bar, speedup in zip(bars, speedups):
            height = bar.get_height()
            ax.annotate(
                f"{round(speedup)}x",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    # Configure axes
    ax.set_yscale("log")
    ax.set_ylabel("Times Slower vs Event-Based", fontsize=18)
    ax.set_xlabel("Number of Spatial Compartments (N)", fontsize=18)
    ax.set_title(
        "Times Slower vs Event-Based",
        fontsize=22,
        fontweight="bold",
    )

    # X-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"N = {n:,}" for n in N_VALUES], fontsize=16)

    # Y-axis tick labels
    ax.tick_params(axis="y", labelsize=14)

    # Grid and legend
    ax.grid(True, axis="y", ls="-", alpha=0.3, which="both")
    ax.legend(fontsize=14, loc="upper left")

    # Set y-axis limits with some padding
    comparison_models = ["HH Cable", "Passive Cable", "Hybrid Wave"]
    max_speedup = max(results[n][m] for n in N_VALUES for m in comparison_models)
    ax.set_ylim(0.5, max_speedup * 2)

    # Final styling
    plt.tight_layout()
    output_path = _OUTPUT_DIR / "speedup_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SPEEDUP COMPARISON BENCHMARK")
    print("Event-Based Model vs Traditional PDE Models")
    print("=" * 60)

    t_ms, va_src = prepare_input_signal()
    results = benchmark(t_ms, va_src)
    plot_speedup_comparison(results)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Times Slower vs Event-Based Model")
    print("=" * 60)
    print(f"{'N':<12} {'HH':<12} {'PC':<12} {'WV':<12}")
    print("-" * 60)
    for n in N_VALUES:
        print(
            f"{n:<12,} {results[n]['HH Cable']:<12.1f}x {results[n]['Passive Cable']:<12.1f}x {results[n]['Hybrid Wave']:<12.1f}x"
        )

    print("\nBenchmark Complete.")
