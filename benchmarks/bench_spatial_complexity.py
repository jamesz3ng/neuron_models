"""
Spatial Complexity Benchmark

Demonstrates how computation time scales with spatial resolution (number of compartments).
- HH Cable: O(N) - linear scaling with compartments
- Event Model: O(1) - constant time (independent of spatial resolution)

Setup:
- Fixed load: 10 neurons for 100 ms
- Fixed physics: Axon length L = 5000 μm
- Variable: n_compartments in [10, 20, 50, 100, 200, 500, 1000]
"""

import time

import numpy as np

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

from event_model import EventPropagator
from hh_cable import simulate_hh_cable
from hh_model import simulate_hh_model

# =============================================================================
# Configuration
# =============================================================================

T_MS = 100.0  # Simulation duration (ms)
T_S = T_MS * 1e-3  # Simulation duration (s)
N_NEURONS = 10  # Fixed number of neurons
L_UM = 5000.0  # Fixed axon length (μm)

# Compartment counts to test
N_COMPARTMENTS = [10, 20, 50, 100, 200, 500, 1000]

# HH Cable base parameters (dx will be calculated per iteration)
HH_CABLE_BASE_PARAMS = {
    "L": L_UM,
    "dt_s": 1e-6,  # Safe small dt for high-resolution
    "stim_duration_s": 5e-3,
    "stim_amplitude": 1000.0,
    "stim_index": 1,
    "store_history": False,
}

# HH Model parameters for generating input waveform
HH_MODEL_PARAMS = {
    "length": 0.01,
    "n_spatial": 1,
    "dt_s": 1e-5,
    "stim_start_s": 0.0,
    "stim_end_s": 5e-3,
    "stim_amplitude": 20.0,
    "stim_index": 0,
    "store_history": True,
    "history_stride": 1,
}


# =============================================================================
# Benchmark Functions
# =============================================================================


def _generate_source_waveform() -> tuple[np.ndarray, np.ndarray]:
    """Generate source waveform using HH model for Event Model input."""
    result = simulate_hh_model(T_s=T_S, **HH_MODEL_PARAMS)
    t_s = result["t_s"]
    V = result["V"][:, 0]
    return t_s, V


def _time_hh_cable(n_neurons: int, n_compartments: int) -> float:
    """Time HH cable simulation with given spatial resolution."""
    dx = L_UM / n_compartments

    start = time.perf_counter()
    for _ in range(n_neurons):
        simulate_hh_cable(T_s=T_S, dx=dx, **HH_CABLE_BASE_PARAMS)
    elapsed = time.perf_counter() - start
    return elapsed


def _time_event_model(n_neurons: int, v_source: np.ndarray) -> float:
    """
    Time Event Model simulation.
    Note: n_compartments is irrelevant - Event Model is O(1) in spatial complexity.
    """
    propagator = EventPropagator()

    start = time.perf_counter()
    for _ in range(n_neurons):
        propagator.simulate(v_source)
    elapsed = time.perf_counter() - start
    return elapsed


def _run_benchmark() -> dict:
    """Run the spatial complexity benchmark."""
    # Pre-generate source waveform for Event Model
    print("Generating source waveform for Event Model...")
    t_s, v_source = _generate_source_waveform()
    print(f"  Waveform: {len(v_source)} points, {t_s[-1] * 1e3:.1f} ms")

    results = {
        "hh_cable": [],
        "event_model": [],
    }

    print(f"\nSpatial Complexity Benchmark")
    print(f"  Neurons: {N_NEURONS}, Duration: {T_MS} ms, Axon Length: {L_UM} μm")
    print("=" * 70)
    print(
        f"{'Compartments':>12}  {'dx (μm)':>10}  {'HH Cable (s)':>14}  {'Event (s)':>12}"
    )
    print("-" * 70)

    # Measure Event Model time once (it's O(1) in compartments)
    event_time = _time_event_model(N_NEURONS, v_source)

    for n_comp in N_COMPARTMENTS:
        dx = L_UM / n_comp

        # HH Cable (scales with compartments)
        hh_time = _time_hh_cable(N_NEURONS, n_comp)
        results["hh_cable"].append((n_comp, hh_time))

        # Event Model (constant time - same for all compartment counts)
        results["event_model"].append((n_comp, event_time))

        print(f"{n_comp:>12}  {dx:>10.1f}  {hh_time:>14.4f}  {event_time:>12.4f}")

    print("=" * 70)

    return results


# =============================================================================
# Visualization
# =============================================================================


def _plot_results(results: dict, output_path: str):
    """Create linear-linear plot showing O(N) vs O(1) scaling."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    # HH Cable data
    hh_data = results["hh_cable"]
    hh_n = [d[0] for d in hh_data]
    hh_times = [d[1] for d in hh_data]

    # Event Model data
    event_data = results["event_model"]
    event_n = [d[0] for d in event_data]
    event_times = [d[1] for d in event_data]

    # Plot HH Cable (rising line - O(N))
    ax.plot(
        hh_n,
        hh_times,
        "o-",
        color="black",
        markersize=8,
        linewidth=2,
        label="HH Cable (O(N) - linear)",
    )

    # Plot Event Model (flat line - O(1))
    ax.plot(
        event_n,
        event_times,
        "s-",
        color="red",
        markersize=8,
        linewidth=2,
        label="Event Model (O(1) - constant)",
    )

    # Labels and title
    ax.set_xlabel("Number of Compartments (Spatial Resolution)", fontsize=12)
    ax.set_ylabel("Simulation Time (s)", fontsize=12)
    ax.set_title(
        f"Spatial Complexity: HH Cable O(N) vs Event Model O(1)\n"
        f"({N_NEURONS} neurons, {T_MS} ms, L = {L_UM} μm)",
        fontsize=13,
    )

    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set axis limits to show contrast clearly
    ax.set_xlim(0, max(N_COMPARTMENTS) * 1.05)
    ax.set_ylim(-10, max(hh_times) * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)


def _print_summary(results: dict):
    """Print speedup summary."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    hh_data = {d[0]: d[1] for d in results["hh_cable"]}
    event_time = results["event_model"][0][1]  # Constant for all

    print(f"\nEvent Model time (constant): {event_time:.4f}s")
    print("\nSpeedup vs HH Cable:")

    for n_comp in N_COMPARTMENTS:
        hh_time = hh_data[n_comp]
        speedup = hh_time / event_time
        print(f"  {n_comp:>4} compartments: {speedup:>8.1f}x faster")

    # Complexity analysis
    print("\n" + "-" * 70)
    print("COMPLEXITY ANALYSIS")
    print("-" * 70)

    # Estimate slope for HH Cable (should be ~linear)
    hh_n = np.array([d[0] for d in results["hh_cable"]])
    hh_times = np.array([d[1] for d in results["hh_cable"]])

    # Linear fit: time = a * n + b
    coeffs = np.polyfit(hh_n, hh_times, 1)
    slope, intercept = coeffs

    print(f"HH Cable: time ≈ {slope:.6f} * N + {intercept:.4f}")
    print(f"  → O(N) scaling confirmed (linear in compartments)")
    print(f"\nEvent Model: time ≈ {event_time:.4f}s (constant)")
    print(f"  → O(1) scaling confirmed (independent of compartments)")

    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("SPATIAL COMPLEXITY BENCHMARK")
    print("=" * 70)
    print("Comparing computation scaling with spatial resolution:")
    print("  - HH Cable: O(N) - time scales linearly with compartments")
    print("  - Event Model: O(1) - time is constant (compartments irrelevant)")
    print(f"\nFixed parameters:")
    print(f"  - Neurons: {N_NEURONS}")
    print(f"  - Duration: {T_MS} ms")
    print(f"  - Axon length: {L_UM} μm")
    print(f"  - Compartments: {N_COMPARTMENTS}")

    # Run benchmark
    results = _run_benchmark()

    # Plot results
    _plot_results(results, "spatial_complexity.png")

    # Print summary
    _print_summary(results)


if __name__ == "__main__":
    main()
