"""
Dual Stimulation Comparison (Soma vs AIS)

Compares depolarisation dynamics when stimulus current is applied to:
1. Soma only (standard case)
2. AIS only
3. Both simultaneously

This reveals the difference in depolarisation times and spike initiation
between compartments, demonstrating the AIS's role as the spike initiation zone.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np


def run_dual_stim_simulation(
    t_end_ms: float,
    i_stim_soma: np.ndarray,
    i_stim_ais: np.ndarray,
    dt_ms: float = 0.002,
    # Scaling Parameters
    g_na_s_scale: float = 1.0,
    g_k_s_scale: float = 1.0,
    g_na_a_scale: float = 1.0,
    g_k_a_scale: float = 1.0,
    g_couple_scale: float = 1.0,
    # Kinetic Scaling
    tau_m_scale_ais: float = 0.3,
    tau_h_scale_ais: float = 1.0,
    tau_n_scale_ais: float = 0.5,
):
    """
    Run a 2-compartment simulation with independent stimuli to Soma and AIS.

    Parameters
    ----------
    t_end_ms : float
        Total simulation time in ms.
    i_stim_soma : np.ndarray
        Stimulus current applied to soma (uA/cm2).
    i_stim_ais : np.ndarray
        Stimulus current applied to AIS (uA/cm2).
    dt_ms : float
        Time step in ms.

    Returns
    -------
    dict with keys:
        t_ms, Vs, Va, i_na_s, i_k_s, i_na_a, i_k_a
    """
    from src.physics import HHPhysics

    n_steps = len(i_stim_soma)
    t_ms = np.arange(n_steps) * dt_ms

    # --- Parameters ---
    # Soma (Standard)
    g_Na_s = 120.0 * g_na_s_scale
    g_K_s = 36.0 * g_k_s_scale
    g_L = 0.3

    # AIS (High Density)
    g_Na_a = 1300.0 * g_na_a_scale
    g_K_a = 300.0 * g_k_a_scale

    # Reversal Potentials
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387

    # Coupling
    g_couple = 2.0 * g_couple_scale  # mS/cm2
    C_m = 1.0  # uF/cm2
    inv_C_m = 1.0 / C_m

    # --- Initialization ---
    v_rest = -65.0
    Vs = v_rest
    Va = v_rest

    # Steady state gates
    m, h, n = HHPhysics.steady_state(v_rest)

    # Soma gates
    ms, hs, ns = m, h, n
    # AIS gates
    ma, ha, na = m, h, n

    # History
    Vs_hist = np.zeros(n_steps)
    Va_hist = np.zeros(n_steps)
    i_na_s_hist = np.zeros(n_steps)
    i_k_s_hist = np.zeros(n_steps)
    i_na_a_hist = np.zeros(n_steps)
    i_k_a_hist = np.zeros(n_steps)

    # Loop
    for i in range(n_steps):
        # 1. Calculate Currents
        # Soma
        I_Na_s = g_Na_s * (ms**3) * hs * (Vs - E_Na)
        I_K_s = g_K_s * (ns**4) * (Vs - E_K)
        I_L_s = g_L * (Vs - E_L)

        # AIS
        I_Na_a = g_Na_a * (ma**3) * ha * (Va - E_Na)
        I_K_a = g_K_a * (na**4) * (Va - E_K)
        I_L_a = g_L * (Va - E_L)

        # Axial Currents
        I_axial_s = g_couple * (Va - Vs)
        I_axial_a = g_couple * (Vs - Va)

        # 2. Update Voltages (stimulus to both compartments)
        dVs = (i_stim_soma[i] - I_Na_s - I_K_s - I_L_s + I_axial_s) * inv_C_m
        dVa = (i_stim_ais[i] - I_Na_a - I_K_a - I_L_a + I_axial_a) * inv_C_m

        Vs += dVs * dt_ms
        Va += dVa * dt_ms

        # 3. Update Gates
        # Soma (Standard Kinetics)
        m_inf, tau_m, h_inf, tau_h, n_inf, tau_n = HHPhysics.get_gate_kinetics(Vs)
        ms += dt_ms * (m_inf - ms) / tau_m
        hs += dt_ms * (h_inf - hs) / tau_h
        ns += dt_ms * (n_inf - ns) / tau_n

        # AIS (Scaled Kinetics)
        m_inf, tau_m, h_inf, tau_h, n_inf, tau_n = HHPhysics.get_gate_kinetics(Va)
        ma += dt_ms * (m_inf - ma) / (tau_m * tau_m_scale_ais)
        ha += dt_ms * (h_inf - ha) / (tau_h * tau_h_scale_ais)
        na += dt_ms * (n_inf - na) / (tau_n * tau_n_scale_ais)

        # Store
        Vs_hist[i] = Vs
        Va_hist[i] = Va
        i_na_s_hist[i] = I_Na_s
        i_k_s_hist[i] = I_K_s
        i_na_a_hist[i] = I_Na_a
        i_k_a_hist[i] = I_K_a

    return {
        "t_ms": t_ms,
        "Vs": Vs_hist,
        "Va": Va_hist,
        "i_na_s": i_na_s_hist,
        "i_k_s": i_k_s_hist,
        "i_na_a": i_na_a_hist,
        "i_k_a": i_k_a_hist,
    }


def find_depolarisation_time(t, V, threshold: float = -20.0) -> float | None:
    """Find the time when voltage first crosses threshold."""
    crossings = np.where(V > threshold)[0]
    if len(crossings) > 0:
        return t[crossings[0]]
    return None


def find_peak_time(t, V) -> tuple[float, float]:
    """Find peak voltage and its time."""
    peak_idx = np.argmax(V)
    return t[peak_idx], V[peak_idx]


def main():
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Dual Stimulation: Soma vs AIS Depolarisation Comparison")
    print("=" * 60)

    # Simulation parameters
    DT_MS = 0.002
    T_END = 20.0
    stim_start = 5.0
    stim_dur = 1.0
    stim_amp = 100.0

    n_pts = int(T_END / DT_MS)

    # Create stimulus arrays
    start_idx = int(stim_start / DT_MS)
    end_idx = int((stim_start + stim_dur) / DT_MS)

    # Case 1: Soma only stimulation
    i_stim_soma_only = np.zeros(n_pts)
    i_stim_soma_only[start_idx:end_idx] = stim_amp
    i_stim_none = np.zeros(n_pts)

    # Case 2: AIS only stimulation
    i_stim_ais_only = np.zeros(n_pts)
    i_stim_ais_only[start_idx:end_idx] = stim_amp

    # Case 3: Both stimulated simultaneously
    i_stim_both = np.zeros(n_pts)
    i_stim_both[start_idx:end_idx] = stim_amp

    print("\nRunning simulations...")

    # Run all three cases
    print("  [1/3] Soma-only stimulation...")
    res_soma = run_dual_stim_simulation(T_END, i_stim_soma_only, i_stim_none, DT_MS)

    print("  [2/3] AIS-only stimulation...")
    res_ais = run_dual_stim_simulation(T_END, i_stim_none, i_stim_ais_only, DT_MS)

    print("  [3/3] Dual stimulation (both)...")
    res_both = run_dual_stim_simulation(T_END, i_stim_both, i_stim_both, DT_MS)

    # Analyse depolarisation times
    print("\n" + "=" * 60)
    print("Depolarisation Analysis (threshold = -20 mV)")
    print("=" * 60)

    threshold = -20.0
    t = res_soma["t_ms"]

    results = []
    for name, res in [
        ("Soma-only stim", res_soma),
        ("AIS-only stim", res_ais),
        ("Dual stim", res_both),
    ]:
        t_dep_soma = find_depolarisation_time(t, res["Vs"], threshold)
        t_dep_ais = find_depolarisation_time(t, res["Va"], threshold)
        t_peak_soma, v_peak_soma = find_peak_time(t, res["Vs"])
        t_peak_ais, v_peak_ais = find_peak_time(t, res["Va"])

        results.append(
            {
                "name": name,
                "t_dep_soma": t_dep_soma,
                "t_dep_ais": t_dep_ais,
                "t_peak_soma": t_peak_soma,
                "t_peak_ais": t_peak_ais,
                "v_peak_soma": v_peak_soma,
                "v_peak_ais": v_peak_ais,
            }
        )

        print(f"\n{name}:")
        print("-" * 40)
        if t_dep_soma is not None and t_dep_ais is not None:
            delta = t_dep_soma - t_dep_ais
            print(f"  Soma depolarisation: {t_dep_soma:.3f} ms")
            print(f"  AIS  depolarisation: {t_dep_ais:.3f} ms")
            print(f"  Delta (Soma - AIS):  {delta:.3f} ms")
            if delta > 0:
                print(f"  -> AIS fires {abs(delta) * 1000:.1f} us BEFORE soma")
            else:
                print(f"  -> Soma fires {abs(delta) * 1000:.1f} us BEFORE AIS")
        else:
            print(f"  Soma depolarisation: {t_dep_soma}")
            print(f"  AIS  depolarisation: {t_dep_ais}")

        print(f"  Soma peak: {v_peak_soma:.1f} mV at {t_peak_soma:.3f} ms")
        print(f"  AIS  peak: {v_peak_ais:.1f} mV at {t_peak_ais:.3f} ms")

    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    plot_data = [
        ("Soma-only Stimulation", res_soma, i_stim_soma_only, i_stim_none),
        ("AIS-only Stimulation", res_ais, i_stim_none, i_stim_ais_only),
        ("Dual Stimulation (Both)", res_both, i_stim_both, i_stim_both),
    ]

    colors_soma = "k"
    colors_ais = "b"

    for row, (title, res, stim_s, stim_a) in enumerate(plot_data):
        ax_v = axes[row, 0]
        ax_i = axes[row, 1]

        # Voltage traces
        ax_v.plot(res["t_ms"], res["Vs"], colors_soma, lw=2, label="Soma")
        ax_v.plot(res["t_ms"], res["Va"], colors_ais, lw=2, label="AIS")
        ax_v.axhline(-20, color="gray", ls="--", lw=1, alpha=0.5, label="Threshold")
        ax_v.set_xlim(4, 12)
        ax_v.set_ylabel("Voltage (mV)")
        ax_v.set_title(title)
        ax_v.legend(loc="upper right")
        ax_v.grid(True, alpha=0.3)

        # Stimulus indicator
        ax_i.plot(res["t_ms"], stim_s, colors_soma, lw=2, label="I_stim Soma")
        ax_i.plot(res["t_ms"], stim_a, colors_ais + "--", lw=2, label="I_stim AIS")
        ax_i.set_xlim(4, 12)
        ax_i.set_ylabel(r"Stimulus ($\mu$A/cm$^2$)")
        ax_i.set_title(f"{title} - Stimulus")
        ax_i.legend(loc="upper right")
        ax_i.grid(True, alpha=0.3)

        if row == 2:
            ax_v.set_xlabel("Time (ms)")
            ax_i.set_xlabel("Time (ms)")

    plt.tight_layout()
    output_file = _OUTPUT_DIR / "dual_stim_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved plot to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key observations:
1. When stimulating SOMA only: AIS still fires first due to higher
   Na+ channel density and faster kinetics at the AIS.

2. When stimulating AIS only: AIS fires rapidly, soma follows via
   axial current coupling.

3. When stimulating BOTH: The AIS still initiates the spike first,
   but the timing difference may be reduced.

This demonstrates the AIS's role as the spike initiation zone,
regardless of where the stimulus is applied.
""")


if __name__ == "__main__":
    main()
