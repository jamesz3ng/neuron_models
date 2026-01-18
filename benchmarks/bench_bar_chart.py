"""
Propagation Speed Benchmark - Bar Chart Comparison

Compares wall-clock time to propagate signals through:
1. HH Cable (100-compartment active cable with full HH dynamics)
2. Passive Cable (100-compartment passive cable)
3. Fast Model (delay-line with refractory filtering)
4. Event Model (PCA-based encode/decode)

All models process the same biological load: 1000ms of 50Hz spike activity.
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

import time
import numpy as np


# =============================================================================
# Generate Common Input Signal (50Hz spike train, 1000ms)
# =============================================================================

def generate_50hz_train(duration_ms: float = 1000.0, dt_ms: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
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

def benchmark_hh_cable(n_iterations: int = 10) -> float:
    """
    Benchmark HH Cable model.
    
    Config: L=5000μm, dx=50μm (100 compartments), dt=0.01ms, T=1000ms
    Uses internal stimulus to generate 50Hz pattern.
    """
    from hh_cable import simulate_hh_cable
    
    # Parameters for 50Hz stimulus (20ms ISI = 50 pulses in 1000ms)
    # HH cable uses seconds for time
    T_s = 1.0  # 1000ms
    dt_s = 1e-5  # 0.01ms
    dx = 50.0  # 50μm spacing -> 100 compartments
    L = 5000.0  # 5000μm length
    
    times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        
        # Run with minimal history storage
        result = simulate_hh_cable(
            L=L,
            T_s=T_s,
            dx=dx,
            dt_s=dt_s,
            stim_duration_s=1.0,  # Continuous stimulus
            stim_amplitude=30.0,
            store_history=False,
        )
        
        t_end = time.perf_counter()
        times.append(t_end - t_start)
    
    return np.mean(times)


def benchmark_passive_cable(v_source: np.ndarray, t_ms: np.ndarray, n_iterations: int = 10) -> float:
    """
    Benchmark Passive Cable model with v_source as boundary condition.
    
    Config: L=5000μm, dx=50μm (100 compartments), dt=0.01ms
    """
    from cable_equation import simulate_cable_equation
    
    T_ms = t_ms[-1]
    dt_ms = t_ms[1] - t_ms[0]
    dx = 50.0
    L = 5000.0
    
    times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        
        result = simulate_cable_equation(
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
    
    return np.mean(times)


def benchmark_fast_model(v_source: np.ndarray, t_ms: np.ndarray, n_iterations: int = 10) -> float:
    """
    Benchmark Fast (delay-line) model.
    
    Propagation time for 5000μm at 100μm/ms = 50ms delay.
    """
    from fast_model import simulate_fast_model
    
    # Delay for 5000μm axon at typical conduction velocity
    delay_ms = 50.0  # 5000μm / 100 μm/ms
    
    times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        
        result = simulate_fast_model(
            v_input=v_source,
            t_ms_input=t_ms,
            delay_ms=delay_ms,
            refractory_period_ms=5.0,
        )
        
        t_end = time.perf_counter()
        times.append(t_end - t_start)
    
    return np.mean(times)


def benchmark_event_model(v_source: np.ndarray, t_ms: np.ndarray, n_iterations: int = 10) -> float:
    """
    Benchmark Event (PCA-based) model.
    
    Includes: spike detection → encoding → queueing → decoding.
    """
    from event_model import EventPropagator
    
    # Same delay as fast model
    delay_ms = 50.0
    prop = EventPropagator(delay_ms=delay_ms)
    
    times = []
    for _ in range(n_iterations):
        t_start = time.perf_counter()
        
        result = prop.simulate(v_source, t_ms)
        
        t_end = time.perf_counter()
        times.append(t_end - t_start)
    
    return np.mean(times)


# =============================================================================
# Main
# =============================================================================

def main():
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("Propagation Speed Benchmark")
    print("=" * 70)
    print("\nConfiguration:")
    print("  Duration: 1000ms")
    print("  Input: 50Hz spike train (~50 spikes)")
    print("  Axon length: 5000μm")
    print("  Cable models: 100 compartments (dx=50μm)")
    print("  Time step: 0.01ms")
    
    # Generate common input signal
    print("\n[1/5] Generating 50Hz spike train...")
    t_ms, v_source = generate_50hz_train(duration_ms=1000.0, dt_ms=0.01)
    n_spikes = np.sum(np.diff((v_source > -20).astype(int)) > 0)
    print(f"       Generated {len(v_source)} samples, {n_spikes} spikes detected")
    
    n_iter = 10
    results = {}
    
    # Benchmark HH Cable
    print(f"\n[2/5] Benchmarking HH Cable ({n_iter} iterations)...")
    try:
        results["HH Cable"] = benchmark_hh_cable(n_iterations=n_iter)
        print(f"       Average time: {results['HH Cable']:.4f}s")
    except Exception as e:
        print(f"       ERROR: {e}")
        results["HH Cable"] = None
    
    # Benchmark Passive Cable
    print(f"\n[3/5] Benchmarking Passive Cable ({n_iter} iterations)...")
    try:
        results["Passive Cable"] = benchmark_passive_cable(v_source, t_ms, n_iterations=n_iter)
        print(f"       Average time: {results['Passive Cable']:.4f}s")
    except Exception as e:
        print(f"       ERROR: {e}")
        results["Passive Cable"] = None
    
    # Benchmark Fast Model
    print(f"\n[4/5] Benchmarking Fast Model ({n_iter} iterations)...")
    try:
        results["Fast (Delay)"] = benchmark_fast_model(v_source, t_ms, n_iterations=n_iter)
        print(f"       Average time: {results['Fast (Delay)']:.6f}s")
    except Exception as e:
        print(f"       ERROR: {e}")
        results["Fast (Delay)"] = None
    
    # Benchmark Event Model
    print(f"\n[5/5] Benchmarking Event Model ({n_iter} iterations)...")
    try:
        results["Event (PCA)"] = benchmark_event_model(v_source, t_ms, n_iterations=n_iter)
        print(f"       Average time: {results['Event (PCA)']:.6f}s")
    except Exception as e:
        print(f"       ERROR: {e}")
        results["Event (PCA)"] = None
    
    # Filter out failed benchmarks
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("\nERROR: No benchmarks succeeded!")
        return
    
    # Calculate speedups relative to HH Cable (or slowest model)
    baseline_name = "HH Cable" if "HH Cable" in valid_results else max(valid_results, key=valid_results.get)
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
        speedup_str = f"{speedup:.0f}x" if speedup >= 1 else f"{1/speedup:.1f}x slower"
        print(f"{name:<20} {t:<15.6f} {speedup_str:<15}")
    
    print("-" * 70)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    
    models = list(valid_results.keys())
    times = [valid_results[m] for m in models]
    
    # Colors: Black for HH Cable, Green for Passive, Blue for Fast, Red for Event
    color_map = {
        "HH Cable": "#2C3E50",       # Dark blue-gray
        "Passive Cable": "#27AE60",   # Green
        "Fast (Delay)": "#3498DB",    # Blue
        "Event (PCA)": "#E74C3C",     # Red
    }
    colors = [color_map.get(m, "#95A5A6") for m in models]
    
    # Create bars
    bars = ax.bar(models, times, color=colors, edgecolor="black", linewidth=1.2)
    
    # Log scale
    ax.set_yscale("log")
    
    # Add speedup annotations above bars
    for bar, model in zip(bars, models):
        height = bar.get_height()
        speedup = speedups[model]
        
        if speedup >= 1000:
            label = f"{speedup/1000:.1f}k×"
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
            color=color_map.get(model, "#2C3E50"),
        )
    
    # Labels and title
    ax.set_ylabel("Time to simulate 1s (seconds)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Axon Propagation Speed Benchmark\n"
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
    plt.savefig("speed_benchmark_bar.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: speed_benchmark_bar.png")
    plt.close()
    
    # Summary
    if "HH Cable" in valid_results and "Event (PCA)" in valid_results:
        event_speedup = valid_results["HH Cable"] / valid_results["Event (PCA)"]
        print(f"\n{'='*70}")
        print(f"KEY RESULT: Event Model is {event_speedup:.0f}× faster than HH Cable")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
