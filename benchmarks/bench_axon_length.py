"""
Axon Length Scaling Benchmark: O(L) vs O(1)

Demonstrates how traditional PDE-based models scale linearly with axon length,
while the Event-Based model remains constant (O(1)).

Models:
1. HH Cable: O(L) - Nonlinear PDE
2. Passive Cable: O(L) - Linear PDE
3. Hybrid Wave: O(L) - Wave Equation
4. Hybrid Event: O(1) - Sparse Reconstruction
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
DT_MS = 0.001          # Fixed 1us time step for stability
DX_UM = 10.0           # Fixed spatial resolution (10 um per node)

# Axon lengths to test (in micrometers)
L_VALUES_UM = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

# Thresholds/Limits
HH_CUTOFF_L = 50000    # Don't run HH for L > 50mm (too slow)
N_ITER = 3             # Average over 3 runs

# Colors
COLORS = {
    "HH Cable": "#c44e52",       # Red
    "Passive Cable": "#55a868",  # Green
    "Hybrid Wave": "#8172b3",    # Purple
    "Hybrid Event": "#4c72b0",   # Blue
}

# =============================================================================
# PRE-CALCULATE TRIGGER SIGNAL
# =============================================================================

print("Pre-calculating source trigger signal (100ms)...")
pulse_times = [10.0, 30.0, 50.0, 70.0, 90.0]  # 5 spikes
# Use AIS simulation to get a realistic driving waveform
i_stim = create_pulse_train(T_DURATION_MS, pulse_times, dt_ms=0.002)  # AIS sim uses 0.002
t_ms_src, vs_src, va_src, *_ = run_2comp_simulation(T_DURATION_MS, i_stim, dt_ms=0.002)

# Resample to 0.001ms dt for the benchmark models
t_ms_1us = np.arange(0, T_DURATION_MS, DT_MS)
va_src_1us = np.interp(t_ms_1us, t_ms_src, va_src)

# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def benchmark():
    results = {
        "HH Cable": [],
        "Passive Cable": [],
        "Hybrid Wave": [],
        "Hybrid Event": [],
    }
    
    # Init Propagator once (O(1) setup)
    prop = EventPropagator(delay_ms=0.0)

    for L_um in L_VALUES_UM:
        L_mm = L_um / 1000.0
        n_nodes = int(L_um / DX_UM)
        print(f"\nBenchmarking L = {L_mm:.1f} mm ({n_nodes} nodes)...")

        # 1. HH Cable (O(L))
        if L_um <= HH_CUTOFF_L:
            print("  - HH Cable...")
            iters = []  
            for _ in range(N_ITER):
                t0 = time.perf_counter()
                simulate_hh_cable(
                    L=L_um, dx=DX_UM, dt_s=DT_MS/1000.0, T_s=T_DURATION_MS/1000.0,
                    stim_waveform=va_src_1us, t_s_stim=t_ms_1us/1000.0,
                    stim_index=0, store_history=False
                )
                iters.append(time.perf_counter() - t0)
            results["HH Cable"].append((L_mm, np.mean(iters)))
        
        # 2. Passive Cable (O(L))
        print("  - Passive Cable...")
        iters = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            simulate_cable_equation(
                L=L_um, dx=DX_UM, dt_ms=DT_MS, T_ms=T_DURATION_MS,
                v_input=va_src_1us, t_ms_input=t_ms_1us,
                store_history=False
            )
            iters.append(time.perf_counter() - t0)
        results["Passive Cable"].append((L_mm, np.mean(iters)))

        # 3. Hybrid Wave (O(L))
        print("  - Hybrid Wave...")
        iters = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            simulate_wave_model(
                L=L_um, nx=n_nodes, dt_s=DT_MS/1000.0, T_s=T_DURATION_MS/1000.0,
                v_input=va_src_1us, t_input=t_ms_1us/1000.0,
                store_history=False
            )
            iters.append(time.perf_counter() - t0)
        results["Hybrid Wave"].append((L_mm, np.mean(iters)))

        # 4. Hybrid Event (O(1))
        # Note: Event model cost is fixed with respect to axon length
        print("  - Hybrid Event (Ours)...")
        iters = []
        for _ in range(N_ITER):
            t0 = time.perf_counter()
            # We simulate the reconstruction at a single destination point
            prop.simulate(va_src_1us, t_ms_1us)
            iters.append(time.perf_counter() - t0)
        results["Hybrid Event"].append((L_mm, np.mean(iters)))

    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(results):
    plt.figure(figsize=(10, 7))
    
    for name, data in results.items():
        if not data:
            continue
        l_vals, t_vals = zip(*data)
        
        # Plot markers and lines
        display_name = "Event Based (Ours)" if name == "Hybrid Event" else name
        plt.loglog(l_vals, t_vals, 'o-', color=COLORS[name], label=display_name, linewidth=2, markersize=8)

    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel("Axon Length (mm)", fontsize=16)
    plt.ylabel("Execution Time (s)", fontsize=16)
    plt.title("Axon Length Scaling Benchmark", fontsize=20, fontweight='bold')
    plt.legend(fontsize=13, loc='center right')
    
    # Final styling
    plt.tight_layout()
    output_path = _OUTPUT_DIR / "axon_length_scaling.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    print("="*60)
    print("AXON LENGTH SCALING BENCHMARK")
    print("="*60)
    results = benchmark()
    plot_results(results)
    print("\nBenchmark Complete.")
