from math import exp as math_exp

import numpy as np

from event_model import EventPropagator

# HH Soma Simulation (with external current input)


def run_hh_soma(
    *,
    i_ext: np.ndarray,
    dt_ms: float = 0.01,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
    v_init: float = -65.0,
) -> np.ndarray:
    """
    Run HH point neuron with external current input.

    Parameters:
        i_ext: External current array (nA)
        dt_ms: Time step (ms)
        g_na_scale: Sodium conductance scaling factor
        g_k_scale: Potassium conductance scaling factor
        v_init: Initial membrane potential (mV)

    Returns:
        v_hist: Membrane potential history (mV)
    """
    # HH parameters
    C_m, g_L, E_Na, E_K, E_L = 1.0, 0.3, 50.0, -77.0, -54.387
    g_Na = 120.0 * g_na_scale
    g_K = 36.0 * g_k_scale

    n_time = len(i_ext)

    # Rate functions
    def rates(v):
        vp = v + 0.001 if abs(v + 40) < 0.001 or abs(v + 55) < 0.001 else v
        am = -0.1 * (vp + 40) / (math_exp(-(vp + 40) / 10) - 1)
        bm = 4.0 * math_exp(-(vp + 65) / 18)
        ah = 0.07 * math_exp(-(vp + 65) / 20)
        bh = 1.0 / (1.0 + math_exp(-(vp + 35) / 10))
        an = -0.01 * (vp + 55) / (math_exp(-(vp + 55) / 10) - 1)
        bn = 0.125 * math_exp(-(vp + 65) / 80)
        return am, bm, ah, bh, an, bn

    # Initial gating variables
    am, bm, ah, bh, an, bn = rates(v_init)
    m, h, n = am / (am + bm), ah / (ah + bh), an / (an + bn)
    v = v_init

    v_hist = np.zeros(n_time)
    v_hist[0] = v

    for i in range(1, n_time):
        am, bm, ah, bh, an, bn = rates(v)
        m += (am * (1 - m) - bm * m) * dt_ms
        h += (ah * (1 - h) - bh * h) * dt_ms
        n += (an * (1 - n) - bn * n) * dt_ms

        i_ion = (
            (g_Na * m**3 * h * (v - E_Na))
            + (g_K * n**4 * (v - E_K))
            + (g_L * (v - E_L))
        )

        v += ((i_ext[i - 1] - i_ion) / C_m) * dt_ms
        v_hist[i] = v

    return v_hist


# =============================================================================
# Synaptic Current Generator
# =============================================================================


def make_synaptic_current(
    v_pre: np.ndarray,
    t_ms: np.ndarray,
    *,
    amplitude: float = 20.0,
    duration_ms: float = 1.0,
    delay_ms: float = 1.0,
    threshold: float = -20.0,
) -> np.ndarray:
    """
    Generate synaptic current from presynaptic voltage.

    Detects spikes in v_pre and creates current pulses.

    Parameters:
        v_pre: Presynaptic voltage trace (mV)
        t_ms: Time axis (ms)
        amplitude: Current pulse amplitude (nA)
        duration_ms: Current pulse duration (ms)
        delay_ms: Synaptic delay (ms)
        threshold: Spike detection threshold (mV)

    Returns:
        i_syn: Synaptic current array (nA)
    """
    dt_ms = t_ms[1] - t_ms[0]
    n_time = len(t_ms)
    i_syn = np.zeros(n_time)

    delay_samples = int(delay_ms / dt_ms)
    duration_samples = int(duration_ms / dt_ms)
    lockout_samples = int(5.0 / dt_ms)  # 5ms refractory

    # Detect spike peaks
    i = 0
    while i < n_time - 1:
        # Threshold crossing (upward)
        if v_pre[i] < threshold <= v_pre[i + 1]:
            # Find peak within 2ms window
            search_end = min(i + int(2.0 / dt_ms), n_time)
            peak_idx = i + np.argmax(v_pre[i:search_end])

            # Create current pulse after delay
            pulse_start = peak_idx + delay_samples
            pulse_end = pulse_start + duration_samples

            if pulse_start < n_time:
                i_syn[pulse_start : min(pulse_end, n_time)] = amplitude

            i = peak_idx + lockout_samples
        else:
            i += 1

    return i_syn


# =============================================================================
# Main Demo
# =============================================================================


def main():
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("3-Neuron Chain Demo: N1 → N2 → N3")
    print("=" * 60)

    # Simulation parameters
    dt_ms = 0.01
    t_end_ms = 80.0
    n_time = int(t_end_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms

    axon_delay_ms = 5.0
    prop = EventPropagator(delay_ms=axon_delay_ms)

    # =========================================================================
    # Neuron 1 (Healthy)
    # =========================================================================
    print("\n[N1] Healthy neuron - external stimulus at t=10ms")

    # External stimulus: current pulse at t=10ms
    i_ext_n1 = np.zeros(n_time)
    stim_start = int(10.0 / dt_ms)
    stim_end = int(11.0 / dt_ms)
    i_ext_n1[stim_start:stim_end] = 30.0  # 30 nA, 1ms

    v_soma_n1 = run_hh_soma(i_ext=i_ext_n1, dt_ms=dt_ms)
    res_n1 = prop.simulate(v_soma_n1, t_ms)
    v_axon_n1 = res_n1["v_out"]

    peak_n1 = np.max(v_soma_n1)
    print(f"    Soma peak: {peak_n1:.1f} mV | Spikes detected: {res_n1['n_spikes']}")

    # =========================================================================
    # Neuron 2 ("Sick" - reduced sodium)
    # =========================================================================
    print("\n[N2] Sick neuron (g_Na × 0.6) - synaptic input from N1")

    i_syn_n2 = make_synaptic_current(v_axon_n1, t_ms)
    v_soma_n2 = run_hh_soma(i_ext=i_syn_n2, dt_ms=dt_ms, g_na_scale=0.6)
    res_n2 = prop.simulate(v_soma_n2, t_ms)
    v_axon_n2 = res_n2["v_out"]

    peak_n2 = np.max(v_soma_n2)
    print(f"    Soma peak: {peak_n2:.1f} mV | Spikes detected: {res_n2['n_spikes']}")

    # =========================================================================
    # Neuron 3 (Healthy)
    # =========================================================================
    print("\n[N3] Healthy neuron - synaptic input from N2")

    i_syn_n3 = make_synaptic_current(v_axon_n2, t_ms)
    v_soma_n3 = run_hh_soma(i_ext=i_syn_n3, dt_ms=dt_ms)
    res_n3 = prop.simulate(v_soma_n3, t_ms)
    v_axon_n3 = res_n3["v_out"]

    peak_n3 = np.max(v_soma_n3)
    print(f"    Soma peak: {peak_n3:.1f} mV | Spikes detected: {res_n3['n_spikes']}")

    # =========================================================================
    # Visualization
    # =========================================================================
    print("\nGenerating plot...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    neurons = [
        ("N1 (Healthy)", v_soma_n1, v_axon_n1, peak_n1),
        ("N2 (Sick: g_Na×0.6)", v_soma_n2, v_axon_n2, peak_n2),
        ("N3 (Healthy)", v_soma_n3, v_axon_n3, peak_n3),
    ]

    for ax, (name, v_soma, v_axon, peak) in zip(axes, neurons):
        ax.plot(t_ms, v_soma, "k-", linewidth=1.5, label="V_soma")
        ax.plot(t_ms, v_axon, "r--", linewidth=1.2, label="V_axon_end")

        # Annotate peak
        peak_idx = np.argmax(v_soma)
        peak_time = t_ms[peak_idx]
        ax.annotate(
            f"Peak: {peak:.1f} mV",
            xy=(peak_time, peak),
            xytext=(peak_time + 5, peak + 10),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="blue"),
            color="blue",
        )

        ax.set_ylabel("Voltage (mV)")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-90, 60)

    axes[-1].set_xlabel("Time (ms)")

    fig.suptitle(
        "3-Neuron Chain: Signal Degradation (N2) and Recovery (N3)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("neuron_chain_demo.png", dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  N1 Peak: {peak_n1:.1f} mV (healthy)")
    print(f"  N2 Peak: {peak_n2:.1f} mV (reduced due to g_Na×0.6)")
    print(f"  N3 Peak: {peak_n3:.1f} mV (recovered)")
    print("=" * 60)


if __name__ == "__main__":
    main()
