"""
AIS Parameter Sweep: Finding Optimal Parameters for Sharp Action Potentials

Varies time constants (tau_m, tau_h, tau_n scaling) and channel densities (g_Na, g_K)
to find parameter combinations that produce sharper APs compared to standard HH.

Primary metric: Rise time (10-90%) - faster rise = sharper AP
Secondary metrics: FWHM, peak amplitude, fall time
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np

from src.ais_simulation import run_2comp_simulation
from src.hh_model import simulate_hh_model


# =============================================================================
# Metrics Calculation
# =============================================================================


def calc_spike_metrics(t: np.ndarray, V: np.ndarray) -> dict:
    """
    Calculate spike shape metrics from a voltage trace.

    Returns dict with:
        - peak_amp: Peak amplitude (mV)
        - rise_time: 10-90% rise time (ms)
        - fall_time: 90-10% fall time (ms)
        - fwhm: Full width at half maximum (ms)
        - dv_dt_max: Maximum rate of rise (mV/ms)
    """
    # Find peak
    peak_idx = np.argmax(V)
    peak_val = V[peak_idx]
    peak_time = t[peak_idx]
    base_val = V[0]  # Resting potential

    # Amplitude above rest
    amplitude = peak_val - base_val

    if amplitude < 20.0:  # No spike detected
        return {
            "peak_amp": peak_val,
            "rise_time": np.nan,
            "fall_time": np.nan,
            "fwhm": np.nan,
            "dv_dt_max": np.nan,
        }

    # Thresholds for rise/fall time
    v10 = base_val + 0.1 * amplitude
    v90 = base_val + 0.9 * amplitude

    # Rise time (10% to 90%)
    # Find first crossing of v10 before peak
    pre_peak = V[:peak_idx]
    t_pre = t[:peak_idx]

    cross_10_up = np.where(pre_peak >= v10)[0]
    cross_90_up = np.where(pre_peak >= v90)[0]

    if len(cross_10_up) > 0 and len(cross_90_up) > 0:
        t10_rise = t_pre[cross_10_up[0]]
        t90_rise = t_pre[cross_90_up[0]]
        rise_time = t90_rise - t10_rise
    else:
        rise_time = np.nan

    # Fall time (90% to 10%)
    post_peak = V[peak_idx:]
    t_post = t[peak_idx:]

    cross_90_down = np.where(post_peak <= v90)[0]
    cross_10_down = np.where(post_peak <= v10)[0]

    if len(cross_90_down) > 0 and len(cross_10_down) > 0:
        t90_fall = t_post[cross_90_down[0]]
        t10_fall = t_post[cross_10_down[0]]
        fall_time = t10_fall - t90_fall
    else:
        fall_time = np.nan

    # FWHM
    half_max = base_val + 0.5 * amplitude
    above_half = V >= half_max
    crossings = np.where(np.diff(above_half.astype(int)))[0]

    if len(crossings) >= 2:
        # Find crossing pair around peak
        left_cross = crossings[crossings < peak_idx]
        right_cross = crossings[crossings > peak_idx]

        if len(left_cross) > 0 and len(right_cross) > 0:
            fwhm = t[right_cross[0]] - t[left_cross[-1]]
        else:
            fwhm = np.nan
    else:
        fwhm = np.nan

    # Max dV/dt
    dt = t[1] - t[0]
    dv_dt = np.gradient(V, dt)
    dv_dt_max = np.max(dv_dt)

    return {
        "peak_amp": peak_val,
        "rise_time": rise_time,
        "fall_time": fall_time,
        "fwhm": fwhm,
        "dv_dt_max": dv_dt_max,
    }


def run_ais_with_params(
    tau_m_scale: float = 0.8,
    tau_h_scale: float = 1.0,
    tau_n_scale: float = 0.5,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run AIS simulation with specified parameters.

    Returns (t, V_soma, V_ais)
    """
    DT_MS = 0.002
    T_END = 20.0
    STIM_START = 5.0
    STIM_DUR = 1.0
    STIM_AMP = 100.0

    n_pts = int(T_END / DT_MS)
    i_stim = np.zeros(n_pts)
    start_idx = int(STIM_START / DT_MS)
    end_idx = int((STIM_START + STIM_DUR) / DT_MS)
    i_stim[start_idx:end_idx] = STIM_AMP

    t, Vs, Va, *_ = run_2comp_simulation(
        T_END,
        i_stim,
        DT_MS,
        tau_m_scale_ais=tau_m_scale,
        tau_h_scale_ais=tau_h_scale,
        tau_n_scale_ais=tau_n_scale,
        g_na_a_scale=g_na_scale,
        g_k_a_scale=g_k_scale,
    )

    return t, Vs, Va


def run_standard_hh() -> tuple[np.ndarray, np.ndarray]:
    """
    Run standard HH point neuron for comparison.

    Returns (t, V)
    """
    result = simulate_hh_model(
        n_spatial=1,
        dt_s=2e-6,  # 0.002 ms
        T_s=20e-3,  # 20 ms
        stim_start_s=5e-3,
        stim_end_s=6e-3,
        stim_amplitude=100.0,
        store_history=True,
    )

    t_ms = result["t_s"] * 1e3
    V = result["V"][:, 0]

    return t_ms, V


# =============================================================================
# Parameter Sweeps
# =============================================================================


def sweep_1d_tau(
    param_name: str, tau_range: np.ndarray
) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    """
    Sweep a single tau parameter while holding others at default.

    Args:
        param_name: One of 'tau_m', 'tau_h', 'tau_n'
        tau_range: Array of scale values to test

    Returns:
        (tau_range, metrics_list, traces_list)
    """
    metrics_list = []
    traces_list = []

    defaults = {"tau_m": 0.8, "tau_h": 1.0, "tau_n": 0.5}

    for tau_val in tau_range:
        params = defaults.copy()
        params[param_name] = tau_val

        t, Vs, Va = run_ais_with_params(
            tau_m_scale=params["tau_m"],
            tau_h_scale=params["tau_h"],
            tau_n_scale=params["tau_n"],
        )

        metrics = calc_spike_metrics(t, Va)
        metrics_list.append(metrics)
        traces_list.append((t, Va))

    return tau_range, metrics_list, traces_list


def sweep_2d_tau_m_n(
    tau_m_range: np.ndarray, tau_n_range: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D sweep of tau_m vs tau_n scaling.

    Returns:
        (tau_m_range, tau_n_range, rise_time_grid)
    """
    rise_time_grid = np.zeros((len(tau_m_range), len(tau_n_range)))
    total = len(tau_m_range) * len(tau_n_range)
    count = 0

    for i, tau_m in enumerate(tau_m_range):
        for j, tau_n in enumerate(tau_n_range):
            t, Vs, Va = run_ais_with_params(tau_m_scale=tau_m, tau_n_scale=tau_n)
            metrics = calc_spike_metrics(t, Va)
            rise_time_grid[i, j] = metrics["rise_time"]
            count += 1
            print(f"    tau sweep: {count}/{total}", end="\r")
    print()

    return tau_m_range, tau_n_range, rise_time_grid


def sweep_2d_conductance(
    g_na_range: np.ndarray, g_k_range: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D sweep of g_Na vs g_K scaling.

    Returns:
        (g_na_range, g_k_range, rise_time_grid)
    """
    rise_time_grid = np.zeros((len(g_na_range), len(g_k_range)))
    total = len(g_na_range) * len(g_k_range)
    count = 0

    for i, g_na in enumerate(g_na_range):
        for j, g_k in enumerate(g_k_range):
            t, Vs, Va = run_ais_with_params(g_na_scale=g_na, g_k_scale=g_k)
            metrics = calc_spike_metrics(t, Va)
            rise_time_grid[i, j] = metrics["rise_time"]
            count += 1
            print(f"    conductance sweep: {count}/{total}", end="\r")
    print()

    return g_na_range, g_k_range, rise_time_grid


# =============================================================================
# Main
# =============================================================================


def main():
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("AIS Parameter Sweep for Sharper Action Potentials")
    print("=" * 60)

    # Define sweep ranges (reduced resolution for speed)
    tau_range = np.linspace(0.2, 1.2, 6)  # 6 points for 1D
    tau_grid_range = np.linspace(0.2, 1.2, 5)  # 5x5 grid
    g_range = np.linspace(0.5, 2.0, 5)  # 5x5 grid

    # -------------------------------------------------------------------------
    # 1D Sweeps
    # -------------------------------------------------------------------------
    print("\n[1/4] Running 1D tau_m sweep...")
    tau_m_vals, tau_m_metrics, tau_m_traces = sweep_1d_tau("tau_m", tau_range)

    print("[2/4] Running 1D tau_h sweep...")
    tau_h_vals, tau_h_metrics, tau_h_traces = sweep_1d_tau("tau_h", tau_range)

    print("[3/4] Running 1D tau_n sweep...")
    tau_n_vals, tau_n_metrics, tau_n_traces = sweep_1d_tau("tau_n", tau_range)

    # -------------------------------------------------------------------------
    # 2D Sweeps
    # -------------------------------------------------------------------------
    print("[4/4] Running 2D sweeps (tau_m vs tau_n, g_Na vs g_K)...")
    tau_m_grid, tau_n_grid, rise_time_tau = sweep_2d_tau_m_n(
        tau_grid_range, tau_grid_range
    )
    g_na_grid, g_k_grid, rise_time_g = sweep_2d_conductance(g_range, g_range)

    # -------------------------------------------------------------------------
    # Find optimal parameters
    # -------------------------------------------------------------------------
    # From tau sweep
    min_idx = np.unravel_index(np.nanargmin(rise_time_tau), rise_time_tau.shape)
    opt_tau_m = tau_m_grid[min_idx[0]]
    opt_tau_n = tau_n_grid[min_idx[1]]
    opt_rise_tau = rise_time_tau[min_idx]

    # From conductance sweep
    min_idx_g = np.unravel_index(np.nanargmin(rise_time_g), rise_time_g.shape)
    opt_g_na = g_na_grid[min_idx_g[0]]
    opt_g_k = g_k_grid[min_idx_g[1]]
    opt_rise_g = rise_time_g[min_idx_g]

    print("\n" + "=" * 60)
    print("OPTIMAL PARAMETERS FOUND")
    print("=" * 60)
    print(f"\nFrom tau sweep (g_Na=1.0, g_K=1.0):")
    print(f"  tau_m_scale = {opt_tau_m:.2f}")
    print(f"  tau_n_scale = {opt_tau_n:.2f}")
    print(f"  Rise time   = {opt_rise_tau:.4f} ms")

    print(f"\nFrom conductance sweep (tau_m=0.8, tau_n=0.5):")
    print(f"  g_Na_scale = {opt_g_na:.2f}")
    print(f"  g_K_scale  = {opt_g_k:.2f}")
    print(f"  Rise time  = {opt_rise_g:.4f} ms")

    # -------------------------------------------------------------------------
    # Reference traces for comparison
    # -------------------------------------------------------------------------
    print("\nGenerating comparison traces...")

    # Standard HH
    t_hh, V_hh = run_standard_hh()
    hh_metrics = calc_spike_metrics(t_hh, V_hh)

    # Default AIS
    t_def, Vs_def, Va_def = run_ais_with_params()
    def_metrics = calc_spike_metrics(t_def, Va_def)

    # Optimized AIS (from tau sweep)
    t_opt, Vs_opt, Va_opt = run_ais_with_params(
        tau_m_scale=opt_tau_m, tau_n_scale=opt_tau_n
    )
    opt_metrics = calc_spike_metrics(t_opt, Va_opt)

    print("\n" + "-" * 40)
    print("COMPARISON (Rise Time 10-90%)")
    print("-" * 40)
    print(f"Standard HH:    {hh_metrics['rise_time']:.4f} ms")
    print(f"Default AIS:    {def_metrics['rise_time']:.4f} ms")
    print(f"Optimized AIS:  {opt_metrics['rise_time']:.4f} ms")
    print(
        f"\nSpeedup vs HH:  {hh_metrics['rise_time'] / opt_metrics['rise_time']:.2f}x faster"
    )

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))

    # Row 1: 1D sweeps
    # -----------------
    ax1 = fig.add_subplot(2, 3, 1)
    rise_times_m = [m["rise_time"] for m in tau_m_metrics]
    ax1.plot(tau_m_vals, rise_times_m, "b-o", linewidth=2, markersize=6)
    ax1.axvline(0.8, color="gray", linestyle="--", alpha=0.5, label="Default")
    ax1.axhline(
        hh_metrics["rise_time"],
        color="red",
        linestyle=":",
        alpha=0.7,
        label="Standard HH",
    )
    ax1.set_xlabel("tau_m_scale_ais")
    ax1.set_ylabel("Rise Time 10-90% (ms)")
    ax1.set_title("Effect of Na+ Activation Speed")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    rise_times_h = [m["rise_time"] for m in tau_h_metrics]
    ax2.plot(tau_h_vals, rise_times_h, "g-o", linewidth=2, markersize=6)
    ax2.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="Default")
    ax2.axhline(
        hh_metrics["rise_time"],
        color="red",
        linestyle=":",
        alpha=0.7,
        label="Standard HH",
    )
    ax2.set_xlabel("tau_h_scale_ais")
    ax2.set_ylabel("Rise Time 10-90% (ms)")
    ax2.set_title("Effect of Na+ Inactivation Speed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    rise_times_n = [m["rise_time"] for m in tau_n_metrics]
    ax3.plot(tau_n_vals, rise_times_n, "m-o", linewidth=2, markersize=6)
    ax3.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Default")
    ax3.axhline(
        hh_metrics["rise_time"],
        color="red",
        linestyle=":",
        alpha=0.7,
        label="Standard HH",
    )
    ax3.set_xlabel("tau_n_scale_ais")
    ax3.set_ylabel("Rise Time 10-90% (ms)")
    ax3.set_title("Effect of K+ Activation Speed")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: 2D heatmaps and comparison
    # ---------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    im1 = ax4.imshow(
        rise_time_tau.T,
        extent=(tau_m_grid[0], tau_m_grid[-1], tau_n_grid[0], tau_n_grid[-1]),
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
    )
    ax4.plot(opt_tau_m, opt_tau_n, "r*", markersize=15, label="Optimal")
    ax4.plot(0.8, 0.5, "wo", markersize=10, markeredgecolor="black", label="Default")
    ax4.set_xlabel("tau_m_scale_ais")
    ax4.set_ylabel("tau_n_scale_ais")
    ax4.set_title("Rise Time: tau_m vs tau_n")
    ax4.legend(loc="upper right")
    plt.colorbar(im1, ax=ax4, label="Rise Time (ms)")

    ax5 = fig.add_subplot(2, 3, 5)
    im2 = ax5.imshow(
        rise_time_g.T,
        extent=(g_na_grid[0], g_na_grid[-1], g_k_grid[0], g_k_grid[-1]),
        origin="lower",
        aspect="auto",
        cmap="viridis_r",
    )
    ax5.plot(opt_g_na, opt_g_k, "r*", markersize=15, label="Optimal")
    ax5.plot(1.0, 1.0, "wo", markersize=10, markeredgecolor="black", label="Default")
    ax5.set_xlabel("g_Na_scale")
    ax5.set_ylabel("g_K_scale")
    ax5.set_title("Rise Time: g_Na vs g_K")
    ax5.legend(loc="upper right")
    plt.colorbar(im2, ax=ax5, label="Rise Time (ms)")

    # Comparison traces
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(
        t_hh,
        V_hh,
        "r-",
        linewidth=2,
        label=f"Standard HH (rise={hh_metrics['rise_time']:.3f}ms)",
    )
    ax6.plot(
        t_def,
        Va_def,
        "b-",
        linewidth=2,
        label=f"Default AIS (rise={def_metrics['rise_time']:.3f}ms)",
    )
    ax6.plot(
        t_opt,
        Va_opt,
        "g-",
        linewidth=2,
        label=f"Optimized AIS (rise={opt_metrics['rise_time']:.3f}ms)",
    )
    ax6.set_xlabel("Time (ms)")
    ax6.set_ylabel("Voltage (mV)")
    ax6.set_title("Action Potential Comparison")
    ax6.set_xlim(4, 12)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = _OUTPUT_DIR / "ais_param_sweep.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved plot to {output_file}")

    # -------------------------------------------------------------------------
    # Additional: Trace overlay for tau_m sweep
    # -------------------------------------------------------------------------
    fig2, (ax_traces, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

    # Select subset of traces to plot (adjusted for 6-point sweep)
    indices_to_plot = [0, 1, 2, 3, 4, 5]  # All 6 points
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0, 1, len(indices_to_plot))]

    for idx, color in zip(indices_to_plot, colors):
        t, Va = tau_m_traces[idx]
        tau_val = tau_m_vals[idx]
        ax_traces.plot(t, Va, color=color, linewidth=1.5, label=f"tau_m={tau_val:.1f}")

    # Add HH reference
    ax_traces.plot(t_hh, V_hh, "r--", linewidth=2, alpha=0.7, label="Standard HH")

    ax_traces.set_xlabel("Time (ms)")
    ax_traces.set_ylabel("Voltage (mV)")
    ax_traces.set_title("AIS Traces: Varying tau_m_scale")
    ax_traces.set_xlim(4, 15)
    ax_traces.legend()
    ax_traces.grid(True, alpha=0.3)

    # Zoomed view on rising phase
    for idx, color in zip(indices_to_plot, colors):
        t, Va = tau_m_traces[idx]
        tau_val = tau_m_vals[idx]
        ax_zoom.plot(t, Va, color=color, linewidth=2, label=f"tau_m={tau_val:.1f}")

    ax_zoom.plot(t_hh, V_hh, "r--", linewidth=2, alpha=0.7, label="Standard HH")
    ax_zoom.set_xlabel("Time (ms)")
    ax_zoom.set_ylabel("Voltage (mV)")
    ax_zoom.set_title("Zoomed: Rising Phase")
    ax_zoom.set_xlim(5.0, 6.5)
    ax_zoom.set_ylim(-70, 50)
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file2 = _OUTPUT_DIR / "ais_tau_m_traces.png"
    plt.savefig(output_file2, dpi=150)
    print(f"Saved trace overlay to {output_file2}")

    plt.show()


if __name__ == "__main__":
    main()
