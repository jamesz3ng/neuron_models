
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

from math import exp as math_exp

import matplotlib.pyplot as plt
import numpy as np

# Import your models
from src.event_model import EventPropagator
from src.physics import HHPhysics
from src.simulation import DT_MS, run_simulation

def run_coupled_simulation(
    t_end_ms: float,
    v_pre_trace: np.ndarray,
    coupling_conductance: float = 0.5,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run HH simulation driven by a Gap Junction (Electrical Coupling).

    Current I = g_coupling * (V_pre(t) - V_post(t))

    Parameters
    ----------
    v_pre_trace : np.ndarray
        Voltage trace of the presynaptic axon (input).
    coupling_conductance : float
        Conductance of the gap junction (mS/cm^2).
        Acts like 1/R. Higher = stronger coupling.
    """
    n_time = int(t_end_ms / DT_MS)
    t_ms = np.arange(n_time) * DT_MS

    # Ensure input trace matches duration
    if len(v_pre_trace) < n_time:
        # Pad with resting potential if input is too short
        pad = np.full(n_time - len(v_pre_trace), HHPhysics.v_rest)
        v_pre_trace = np.concatenate([v_pre_trace, pad])

    # HH Parameters
    C_m = HHPhysics.C_m
    g_Na = HHPhysics.g_Na * g_na_scale
    g_K = HHPhysics.g_K * g_k_scale
    g_L = HHPhysics.g_L
    E_Na, E_K, E_L = HHPhysics.E_Na, HHPhysics.E_K, HHPhysics.E_L
    inv_C_m = 1.0 / C_m

    # Initialization
    V = HHPhysics.v_rest
    m, h, n = HHPhysics.steady_state(V)

    V_hist = np.zeros(n_time)
    V_hist[0] = V

    # Integration Loop
    for i in range(1, n_time):
        # 1. Update Gates
        am, bm, ah, bh, an, bn = HHPhysics.rates(V)
        m += (am * (1 - m) - bm * m) * DT_MS
        h += (ah * (1 - h) - bh * h) * DT_MS
        n += (an * (1 - n) - bn * n) * DT_MS

        # 2. Calculate Ionic Currents
        I_Na = g_Na * m**3 * h * (V - E_Na)
        I_K = g_K * n**4 * (V - E_K)
        I_L = g_L * (V - E_L)

        # 3. Calculate Synaptic Current (Gap Junction)
        # I = (V_pre - V_post) / R  ->  I = g * (V_pre - V)
        V_pre = v_pre_trace[i - 1]
        I_syn = coupling_conductance * (V_pre - V)

        # 4. Update Voltage
        # Note: I_syn is ADDED (positive current enters cell)
        dV = (I_syn - I_Na - I_K - I_L) * inv_C_m
        V += dV * DT_MS

        V_hist[i] = V

    return t_ms, V_hist


# Helper for the first neuron (External Stimulus)
def run_external_simulation(t_end_ms, stim_start=10.0, stim_amp=30.0):
    """Standard HH run with square pulse."""
    n_time = int(t_end_ms / DT_MS)
    i_stim = np.zeros(n_time)
    start_idx = int(stim_start / DT_MS)
    end_idx = int((stim_start + 1.0) / DT_MS)
    i_stim[start_idx:end_idx] = stim_amp


    t_ms, V = run_simulation(t_end_ms, i_stim)
    return t_ms, V


# =============================================================================
# Main Demo
# ============================================================================= 


def main():
    print("=" * 60)
    print("3-Neuron Chain: Gap Junction Coupling")
    print("=" * 60)

    # Config
    t_end = 60.0
    delay = 7
    prop = EventPropagator(delay_ms=delay)

    # Tuning the Gap Junction (Conductance in mS/cm^2)
    # Needs to be high enough to transfer the spike
    G_COUPLE = 0.5

    # Neuron 1: The Source (Healthy)
    print("[N1] Firing...")
    t_ms, v_soma1 = run_external_simulation(t_end, stim_start=5.0)

    # Propagate down Axon 1
    res1 = prop.simulate(v_soma1, t_ms)
    v_axon1 = res1["v_out"]
    print(v_axon1, len(v_axon1))
    # Neuron 2: 
    print("[N2] Receiving input from Axon 1...")
    _, v_soma2 = run_coupled_simulation(
        t_end, v_pre_trace=v_axon1, coupling_conductance=G_COUPLE, g_na_scale=1.0
    )
    # Propagate down Axon 2 (This spike should be smaller!)
    res2 = prop.simulate(v_soma2, t_ms)
    v_axon2 = res2["v_out"]
    print(v_axon2, len(v_axon1))

    # Neuron 3: The Recovery 
    
    print("[N3] Receiving input from Axon 2...")
    _, v_soma3 = run_coupled_simulation(
        t_end, v_pre_trace=v_axon2, coupling_conductance=G_COUPLE, g_na_scale=1.0
    )

    # Visualization
    print("Plotting...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot N1
    axes[0].plot(t_ms, v_soma1, "k", label="Soma")
    axes[0].plot(t_ms, v_axon1, "r--", label="Axon End")
    axes[0].set_title("Neuron 1 (Healthy Source)")
    axes[0].legend(loc="upper right")

    # Plot N2
    axes[1].plot(t_ms, v_soma2, "k", label="Soma")
    axes[1].plot(t_ms, v_axon2, "r--", label="Axon End")
    axes[1].set_title(f"Neuron 2 - Peak: {np.max(v_soma2):.1f}mV")
    axes[1].legend(loc="upper right")

    # Plot N3
    axes[2].plot(t_ms, v_soma3, "k", label="Soma")
    axes[2].set_title(f"Neuron 3 - Peak: {np.max(v_soma3):.1f}mV")
    axes[2].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Voltage (mV)")
        ax.set_ylim(-80, 50)

    axes[-1].set_xlabel("Time (ms)")

    plt.tight_layout()
    output_file = _OUTPUT_DIR / "chain_demo_gap.png"
    plt.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
