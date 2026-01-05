"""
Neuron Scaling Benchmark

Measures simulation time as we scale the number of neurons (N).
Demonstrates that Fast Model and Wave Model are orders of magnitude faster than HH Cable.

Models tested:
- hh_cable: Full Hodgkin-Huxley cable equation (physics baseline)
- fast: Pre-generated HH waveform + delay-line propagation
- hybrid_fast: Fresh HH generation + delay-line propagation (realistic cost)
- cable: Pre-generated HH waveform + passive cable propagation
- wave: Pre-generated HH waveform + 1D wave equation propagation
- hybrid_wave: Fresh HH generation + 1D wave equation propagation
"""

import time
import numpy as np

from hh_cable import simulate_hh_cable
from hh_model import simulate_hh_model
from fast_model import simulate_fast_model
from cable_equation import simulate_cable_equation
from wave_model import simulate_wave_model

# =============================================================================
# Configuration
# =============================================================================

T_MS = 100.0  # Simulation duration (ms)
T_S = T_MS * 1e-3  # Simulation duration (s)

# HH Cable optimized parameters
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

# HH Model parameters (point neuron for waveform generation)
HH_MODEL_PARAMS = {
    "length": 0.01,  # Small cable (effectively point neuron)
    "n_spatial": 1,  # Single compartment
    "dt_s": 1e-5,  # s (0.01 ms)
    "stim_start_s": 0.0,
    "stim_end_s": 5e-3,  # 5 ms stimulus
    "stim_amplitude": 20.0,
    "stim_index": 0,
    "store_history": True,  # Need history for waveform output
    "history_stride": 1,
}

# Passive cable parameters
CABLE_PARAMS = {
    "L": 5000.0,  # μm (same as HH cable)
    "dx": 50.0,  # μm
    "dt_ms": 0.01,  # ms (same as HH cable dt)
    "tau_ms": 10.0,
    "lam": 200.0,
    "store_history": False,
}

# Fast model parameters
FAST_MODEL_DELAY_MS = 10.0  # Fixed delay

# Wave model parameters
L_UM = 5000.0  # Original cable length in μm (same as HH cable)
FAST_MODEL_INPUT_X_UM = 1500.0  # Position where clean AP is extracted
RECORD_X_UM = 2500.0  # Recording position (midpoint)
CONDUCTION_VELOCITY = 100.0  # μm/ms
WAVE_MODEL_NX = 200  # Spatial grid points for wave model

# Scaling test parameters
N_COUNTS = [1, 10, 50, 100, 500, 1000, 5000, 10000]
TIMEOUT_SECONDS = 5.0  # Skip larger N if previous took longer than this


# =============================================================================
# Pre-generate standard HH waveform (for fast and cable models)
# =============================================================================


def _generate_hh_waveform() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a standard HH waveform using simulate_hh_model.
    Used as input for 'fast' and 'cable' models.
    """
    result = simulate_hh_model(
        T_s=T_S,
        **HH_MODEL_PARAMS,
    )
    t_s = result["t_s"]
    V = result["V"][:, 0]  # Single compartment
    t_ms = t_s * 1e3
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
    """
    Time running fast_model simulation for N neurons sequentially.
    Uses pre-generated waveform.
    """
    start = time.perf_counter()
    for _ in range(n_neurons):
        simulate_fast_model(
            v_input=v_input,
            t_ms_input=t_ms_input,
            delay_ms=FAST_MODEL_DELAY_MS,
        )
    elapsed = time.perf_counter() - start
    return elapsed


def _time_hybrid_fast(n_neurons: int) -> float:
    """
    Time running hybrid model for N neurons sequentially.
    Each iteration: fresh HH generation + fast_model delay.
    """
    start = time.perf_counter()
    for _ in range(n_neurons):
        # Step A: Generate fresh waveform with HH model
        result = simulate_hh_model(
            T_s=T_S,
            **HH_MODEL_PARAMS,
        )
        t_s = result["t_s"]
        V = result["V"][:, 0]
        t_ms = t_s * 1e3

        # Step B: Propagate with fast_model
        simulate_fast_model(
            v_input=V,
            t_ms_input=t_ms,
            delay_ms=FAST_MODEL_DELAY_MS,
        )
    elapsed = time.perf_counter() - start
    return elapsed


def _time_cable(n_neurons: int, t_ms_input: np.ndarray, v_input: np.ndarray) -> float:
    """
    Time running passive cable simulation for N neurons sequentially.
    Uses pre-generated waveform as input at x=0.
    """
    start = time.perf_counter()
    for _ in range(n_neurons):
        simulate_cable_equation(
            T_ms=T_MS,
            v_input=v_input,
            t_ms_input=t_ms_input,
            **CABLE_PARAMS,
        )
    elapsed = time.perf_counter() - start
    return elapsed


def _get_wave_model_params(t_ms_input: np.ndarray, v_input: np.ndarray) -> dict:
    """
    Calculate wave model parameters for consistent propagation.

    Returns dict with parameters needed for simulate_wave_model.
    """
    # The wave needs to travel from source_x to record_x
    propagation_distance_um = RECORD_X_UM - FAST_MODEL_INPUT_X_UM

    # Use normalized units where L_wave = 1.0
    L_wave = 1.0

    # Recording position in wave model as fraction of cable length
    record_x_wave = propagation_distance_um / L_UM * L_wave

    # Wave velocity: match CONDUCTION_VELOCITY
    travel_time_s = propagation_distance_um / CONDUCTION_VELOCITY * 1e-3
    c_wave = record_x_wave / travel_time_s if travel_time_s > 0 else 20.0

    # Convert time to seconds
    t_s_input = t_ms_input * 1e-3

    return {
        "L": L_wave,
        "c": c_wave,
        "T_s": T_S,
        "nx": WAVE_MODEL_NX,
        "store_history": False,  # Don't store for speed benchmark
        "v_input": v_input,
        "t_input": t_s_input,
        "v_init": v_input[0],
    }


def _time_wave_model(
    n_neurons: int, t_ms_input: np.ndarray, v_input: np.ndarray
) -> float:
    """
    Time running wave_model simulation for N neurons sequentially.
    Uses pre-generated waveform as input.
    """
    wave_params = _get_wave_model_params(t_ms_input, v_input)

    start = time.perf_counter()
    for _ in range(n_neurons):
        simulate_wave_model(**wave_params)
    elapsed = time.perf_counter() - start
    return elapsed


def _time_hybrid_wave(n_neurons: int) -> float:
    """
    Time running hybrid wave model for N neurons sequentially.
    Each iteration: fresh HH generation + wave propagation.
    """
    start = time.perf_counter()
    for _ in range(n_neurons):
        # Step A: Generate fresh waveform with HH model
        result = simulate_hh_model(
            T_s=T_S,
            **HH_MODEL_PARAMS,
        )
        t_s = result["t_s"]
        V = result["V"][:, 0]
        t_ms = t_s * 1e3

        # Step B: Calculate wave parameters and propagate
        wave_params = _get_wave_model_params(t_ms, V)
        simulate_wave_model(**wave_params)
    elapsed = time.perf_counter() - start
    return elapsed


def _run_scaling_benchmark() -> dict:
    """
    Run the scaling benchmark for all models.

    Returns dict mapping model_name -> list of (n, time) tuples
    """
    # Pre-generate standard HH waveform for fast, cable, and wave models
    print("Generating standard HH waveform for 'fast', 'cable', and 'wave' models...")
    t_ms_input, v_input = _generate_hh_waveform()
    print(f"  Waveform: {len(v_input)} points, {t_ms_input[-1]:.1f} ms")

    results = {
        "hh_cable": [],
        "fast": [],
        "hybrid_fast": [],
        "cable": [],
        "wave": [],
        "hybrid_wave": [],
    }

    # Track if we should skip due to timeout
    skip = {
        "hh_cable": False,
        "fast": False,
        "hybrid_fast": False,
        "cable": False,
        "wave": False,
        "hybrid_wave": False,
    }

    print(f"\nScaling Benchmark: T = {T_MS} ms, timeout = {TIMEOUT_SECONDS}s")
    print("=" * 120)
    header = f"{'N':>6}  {'HH Cable':>12}  {'Fast':>12}  {'Hybrid':>12}  {'Cable':>12}  {'Wave':>12}  {'Hyb-Wave':>12}"
    print(header)
    print("-" * 120)

    for n in N_COUNTS:
        row = f"{n:>6}  "
        times = {}

        # HH Cable (physics baseline)
        if not skip["hh_cable"]:
            t = _time_hh_cable(n)
            results["hh_cable"].append((n, t))
            times["hh_cable"] = t
            row += f"{t:>12.4f}  "
            if t > TIMEOUT_SECONDS:
                skip["hh_cable"] = True
        else:
            row += f"{'skip':>12}  "

        # Fast (pre-generated waveform + delay)
        if not skip["fast"]:
            t = _time_fast_model(n, t_ms_input, v_input)
            results["fast"].append((n, t))
            times["fast"] = t
            row += f"{t:>12.6f}  "
            if t > TIMEOUT_SECONDS:
                skip["fast"] = True
        else:
            row += f"{'skip':>12}  "

        # Hybrid Fast (fresh HH generation + delay)
        if not skip["hybrid_fast"]:
            t = _time_hybrid_fast(n)
            results["hybrid_fast"].append((n, t))
            times["hybrid_fast"] = t
            row += f"{t:>12.4f}  "
            if t > TIMEOUT_SECONDS:
                skip["hybrid_fast"] = True
        else:
            row += f"{'skip':>12}  "

        # Cable (pre-generated waveform + passive propagation)
        if not skip["cable"]:
            t = _time_cable(n, t_ms_input, v_input)
            results["cable"].append((n, t))
            times["cable"] = t
            row += f"{t:>12.4f}  "
            if t > TIMEOUT_SECONDS:
                skip["cable"] = True
        else:
            row += f"{'skip':>12}  "

        # Wave (pre-generated waveform + wave propagation)
        if not skip["wave"]:
            t = _time_wave_model(n, t_ms_input, v_input)
            results["wave"].append((n, t))
            times["wave"] = t
            row += f"{t:>12.4f}  "
            if t > TIMEOUT_SECONDS:
                skip["wave"] = True
        else:
            row += f"{'skip':>12}  "

        # Hybrid Wave (fresh HH generation + wave propagation)
        if not skip["hybrid_wave"]:
            t = _time_hybrid_wave(n)
            results["hybrid_wave"].append((n, t))
            times["hybrid_wave"] = t
            row += f"{t:>12.4f}  "
            if t > TIMEOUT_SECONDS:
                skip["hybrid_wave"] = True
        else:
            row += f"{'skip':>12}  "

        print(row, flush=True)

    print("=" * 120)

    return results


# =============================================================================
# Visualization
# =============================================================================


def _plot_scaling(results: dict, output_path: str):
    """Create log-log scaling plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot styling
    styles = {
        "hh_cable": {
            "color": "black",
            "marker": "o",
            "markersize": 8,
            "linewidth": 2,
            "label": "HH Cable (full physics)",
        },
        "fast": {
            "color": "red",
            "marker": "s",
            "markersize": 8,
            "linewidth": 2,
            "label": "Fast (pre-gen + delay)",
        },
        "hybrid_fast": {
            "color": "blue",
            "marker": "^",
            "markersize": 8,
            "linewidth": 2,
            "label": "Hybrid Fast (HH gen + delay)",
        },
        "cable": {
            "color": "green",
            "marker": "d",
            "markersize": 8,
            "linewidth": 2,
            "label": "Cable (pre-gen + passive)",
        },
        "wave": {
            "color": "purple",
            "marker": "v",
            "markersize": 8,
            "linewidth": 2,
            "label": "Wave (pre-gen + wave eq.)",
        },
        "hybrid_wave": {
            "color": "orange",
            "marker": "p",
            "markersize": 8,
            "linewidth": 2,
            "label": "Hybrid Wave (HH gen + wave eq.)",
        },
    }

    for model_name, data in results.items():
        if not data:
            continue
        n_values = [d[0] for d in data]
        times = [d[1] for d in data]
        style = styles.get(model_name, {})
        ax.loglog(n_values, times, **style)

    # Add reference line for O(N)
    n_ref = np.array([1, 10000])
    ax.loglog(
        n_ref,
        n_ref * 1e-5,
        "gray",
        linestyle="--",
        alpha=0.5,
        linewidth=1,
        label="O(N) reference",
    )

    # Add timeout line
    ax.axhline(TIMEOUT_SECONDS, color="orange", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.text(
        1.5,
        TIMEOUT_SECONDS * 1.2,
        f"Timeout ({TIMEOUT_SECONDS}s)",
        fontsize=9,
        color="orange",
    )

    ax.set_xlabel("Number of Neurons (N)", fontsize=12)
    ax.set_ylabel("Simulation Time (seconds)", fontsize=12)
    ax.set_title(f"Neuron Scaling Benchmark (T = {T_MS} ms simulation)", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)


def _print_summary(results: dict):
    """Print speedup summary."""
    print("\n" + "=" * 70)
    print("SPEEDUP SUMMARY (vs HH Cable)")
    print("=" * 70)

    hh_data = {d[0]: d[1] for d in results.get("hh_cable", [])}

    if not hh_data:
        print("No HH Cable data for comparison.")
        return

    for model_name in ["fast", "hybrid_fast", "cable", "wave", "hybrid_wave"]:
        model_data = {d[0]: d[1] for d in results.get(model_name, [])}
        shared_ns = sorted(set(hh_data.keys()) & set(model_data.keys()))

        if not shared_ns:
            print(f"\n{model_name}: No shared N values")
            continue

        print(f"\n{model_name}:")
        for n in shared_ns:
            speedup = hh_data[n] / model_data[n]
            print(f"  N={n:>5}: {speedup:>10.1f}x faster")

    # Find largest N for each model and project HH cable time
    print("\n" + "-" * 70)
    print("PROJECTIONS (extrapolating HH Cable time)")
    print("-" * 70)

    # Estimate HH time per neuron from available data
    if len(results["hh_cable"]) >= 1:
        last_hh = results["hh_cable"][-1]
        hh_time_per_neuron = last_hh[1] / last_hh[0]

        for model_name in ["fast", "hybrid_fast", "cable", "wave", "hybrid_wave"]:
            model_data = results.get(model_name, [])
            if model_data:
                largest_n = model_data[-1][0]
                model_time = model_data[-1][1]
                projected_hh = hh_time_per_neuron * largest_n

                print(f"\nAt N = {largest_n}:")
                print(f"  {model_name}: {model_time:.4f}s (actual)")
                print(
                    f"  HH Cable: {projected_hh:.1f}s = {projected_hh / 60:.1f} min (projected)"
                )
                print(f"  Speedup: {projected_hh / model_time:.0f}x")

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("NEURON SCALING BENCHMARK")
    print("=" * 70)
    print("Models:")
    print("  - hh_cable:     Full HH cable equation (physics baseline)")
    print("  - fast:         Pre-generated waveform + delay-line")
    print("  - hybrid_fast:  Fresh HH generation + delay-line")
    print("  - cable:        Pre-generated waveform + passive cable")
    print("  - wave:         Pre-generated waveform + 1D wave equation")
    print("  - hybrid_wave:  Fresh HH generation + 1D wave equation")
    print(f"\nDuration: {T_MS} ms per simulation")
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
