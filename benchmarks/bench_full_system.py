"""
Full System Performance Benchmark

Compares total execution time for three architectural approaches:
1. HH Cable (Standard): 100-compartment Hodgkin-Huxley cable.
2. Hybrid Wave: Soma/AIS physics (2-comp) -> Wave propagation (100 nodes).
3. Hybrid Event: Soma/AIS physics (2-comp) -> Event encoding -> Reconstruction.

Isolates Generation Time vs Propagation Time.
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
from src.ais_simulation import run_2comp_simulation
from src.wave_model import simulate_wave_model
from src.event_model import EventPropagator
from src.simulation import create_pulse_train

# =============================================================================
# CONFIGURATION
# =============================================================================

T_END_MS = 1000.0
FREQ_HZ = 50.0
ISI_MS = 1000.0 / FREQ_HZ
N_ITER = 3  # Reduced iterations as HH Cable with 10k nodes is slow

# Fixed Axon Parameters
AXON_L_UM = 5000.0
NX = 10000  # 10,000 nodes
DX_UM = AXON_L_UM / NX
CONDUCTION_VELOCITY_M_S = 10.0 # m/s
DELAY_MS = (AXON_L_UM / 1000.0) / (CONDUCTION_VELOCITY_M_S / 1000.0) # 50ms

# Standard Colors
COLOR_GEN = "#8c8c8c"  # Neutral Gray (Soma/Generation)
COLOR_PROP = "#4c72b0" # Deep Blue (Axon/Propagation)
COLOR_MIXED = "#55a868" # Deep Green (HH Cable)

# =============================================================================
# PREPARE STIMULUS
# =============================================================================

pulse_times = [10.0 + i * ISI_MS for i in range(int(T_END_MS / ISI_MS))]
i_stim_soma_002 = create_pulse_train(T_END_MS, pulse_times, dt_ms=0.002)
i_stim_soma_010 = create_pulse_train(T_END_MS, pulse_times, dt_ms=0.010)

# =============================================================================
# BENCHMARK ROUTINES
# =============================================================================

def bench_hh_cable():
    """HH Cable: Fully active 100-compartment cable."""
    print(f"  [HH Cable] Benchmarking {N_ITER} iterations...")
    
    # We use same dt as generation (0.002ms) for fair comparison
    t_s_stim = np.arange(len(i_stim_soma_002)) * 0.000002 # 0.002ms in s
    
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        # Use no history for performance benchmarking
        simulate_hh_cable(
            L=AXON_L_UM,
            T_s=T_END_MS / 1000.0,
            dx=DX_UM,
            dt_s=2e-6, # 0.002ms
            stim_waveform=i_stim_soma_002,
            t_s_stim=t_s_stim,
            stim_index=0,
            store_history=False
        )
        times.append(time.perf_counter() - t0)
    
    return np.mean(times), 0.0, np.mean(times) # (total, gen, prop) - Mixed

def bench_hybrid_wave():
    """Hybrid Wave: 2-comp Soma/AIS -> 100-node Wave."""
    print(f"  [Hybrid Wave] Benchmarking {N_ITER} iterations...")
    
    gen_times = []
    prop_times = []
    
    for _ in range(N_ITER):
        # Step A: Generation
        t0 = time.perf_counter()
        t_ms, vs, va, *_ = run_2comp_simulation(T_END_MS, i_stim_soma_002, dt_ms=0.002)
        gen_times.append(time.perf_counter() - t0)
        
        # Step B: Propagation
        t1 = time.perf_counter()
        simulate_wave_model(
            L=AXON_L_UM,
            c=CONDUCTION_VELOCITY_M_S,
            T_s=T_END_MS / 1000.0,
            nx=NX,
            v_input=va, # Drive by AIS trace
            t_input=t_ms / 1000.0,
            store_history=False
        )
        prop_times.append(time.perf_counter() - t1)
        
    return np.mean(gen_times) + np.mean(prop_times), np.mean(gen_times), np.mean(prop_times)

def bench_hybrid_event():
    """Hybrid Event: 2-comp Soma/AIS -> PCA EventPropagator."""
    print(f"  [Hybrid Event] Benchmarking {N_ITER} iterations...")
    
    # Init propagator once
    prop = EventPropagator(delay_ms=DELAY_MS)
    
    gen_times = []
    prop_times = []
    
    for _ in range(N_ITER):
        # Step A: Generation
        t0 = time.perf_counter()
        t_ms, vs, va, *_ = run_2comp_simulation(T_END_MS, i_stim_soma_002, dt_ms=0.002)
        gen_times.append(time.perf_counter() - t0)
        
        # Step B: Propagation
        t1 = time.perf_counter()
        prop.simulate(va, t_ms)
        prop_times.append(time.perf_counter() - t1)
        
    return np.mean(gen_times) + np.mean(prop_times), np.mean(gen_times), np.mean(prop_times)

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("FULL SYSTEM PERFORMANCE BENCHMARK")
    print(f"Simulating 1000ms activity @ 50Hz, {NX} nodes")
    print("=" * 70)

    results = {}
    
    # 1. Hybrid Event (Ours)
    total, gen, prop = bench_hybrid_event()
    results["Hybrid Event (Ours)"] = {"gen": gen, "prop": prop, "mixed": 0, "total": total}

    # 2. Hybrid Wave
    total, gen, prop = bench_hybrid_wave()
    results["Hybrid Wave"] = {"gen": gen, "prop": prop, "mixed": 0, "total": total}

    # 3. HH Cable
    total, gen, prop = bench_hh_cable()
    results["HH Cable (10,000 nodes)"] = {"gen": 0, "prop": 0, "mixed": total, "total": total}

    # --- Print Results ---
    print("\n" + "-" * 70)
    print(f"{'System':<20} {'Gen (ms)':<12} {'Prop (ms)':<12} {'Total (ms)':<12}")
    print("-" * 70)
    for name, data in results.items():
        total = (data["gen"] + data["prop"] + data["mixed"]) * 1000
        gen = data["gen"] * 1000
        prop = data["prop"] * 1000
        mixed = data["mixed"] * 1000
        
        gen_str = f"{gen:.1f}" if gen > 0 else "-"
        prop_str = f"{prop:.1f}" if prop > 0 else "-"
        if data["mixed"] > 0:
            gen_str = "-"
            prop_str = "-"
            
        print(f"{name:<20} {gen_str:<12} {prop_str:<12} {total:<12.1f}")
    print("-" * 70)

    # --- Visualization ---
    systems = list(results.keys())
    total_means = [results[s]["total"] * 1000 for s in systems]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Single bar plot
    ax.bar(systems, total_means, color=[COLOR_PROP, COLOR_GEN, COLOR_MIXED], edgecolor="black", alpha=0.9)

    ax.set_ylabel("Execution Time (ms)", fontsize=12)
    ax.set_title(f"Full System Latency (1000ms sim duration, {NX} nodes)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Use log scale if differences are massive
    if max(total_means) / min(total_means) > 10:
        ax.set_yscale("log")
        ax.set_ylabel("Execution Time (ms, log scale)")

    plt.tight_layout()
    output_path = _OUTPUT_DIR / "bench_full_system.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nBenchmark visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
