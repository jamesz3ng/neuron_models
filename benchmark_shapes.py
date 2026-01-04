"""
AP Shape Comparison Benchmark

Compares action potential shapes across models:
- hh_cable.py (gold standard baseline) - recorded at midpoint
- hh_model.py - recorded at stimulus site (x=0) due to propagation limitations
- fast_model.py (delay-line using hh_cable waveform as input)

Units: μm for length, ms for time, mV for voltage.

NOTE: hh_model.py has a diffusion coefficient bug that prevents proper AP propagation.
      For fair shape comparison, we record hh_model at the stimulus site where the AP
      is generated, while hh_cable is recorded at the midpoint (propagated AP).
"""

import numpy as np

from hh_cable import simulate_hh_cable
from hh_model import simulate_hh_model
from fast_model import simulate_fast_model

# =============================================================================
# Configuration (hardcoded defaults)
# =============================================================================

L_UM = 5000.0  # Axon length in μm
RECORD_X_UM = 2500.0  # Recording position (midpoint)
FAST_MODEL_INPUT_X_UM = 1500.0  # Position to extract clean AP for fast_model input
T_MS = 40.0  # Simulation duration in ms
BASELINE_MV = -70.0  # Target baseline for alignment
STIM_AMPLITUDE_CABLE = 1000.0  # Stimulus amplitude for hh_cable (μA/cm² equivalent)
STIM_AMPLITUDE_MODEL = (
    20.0  # Stimulus amplitude for hh_model (nA/mm² - different units)
)
STIM_DURATION_MS = 5.0  # Stimulus duration in ms
CONDUCTION_VELOCITY = 100.0  # μm/ms for fast_model
RMSE_WINDOW_MS = 10.0  # ±10ms window for RMSE calculation
# Spatial and time steps - balanced for speed and stability
DX_UM = 50.0  # 50 μm spatial step (coarser but faster)
DT_S = 1e-5  # 0.01 ms time step (stable with dx=50)
HISTORY_STRIDE = 10  # Downsample history


# =============================================================================
# Simulation Runners
# =============================================================================


def _run_hh_cable(L_um: float, T_ms: float, record_x_um: float) -> dict:
    """
    Run hh_cable simulation and extract V(t) at recording position.
    hh_cable uses μm natively.

    Returns dict with:
        - t_ms: time array in ms
        - V_record: voltage at recording position (midpoint)
        - V_source: voltage at FAST_MODEL_INPUT_X_UM for clean AP input to fast_model
    """
    T_s = T_ms * 1e-3
    stim_duration_s = STIM_DURATION_MS * 1e-3

    result = simulate_hh_cable(
        L=L_um,
        T_s=T_s,
        dx=DX_UM,
        dt_s=DT_S,
        stim_duration_s=stim_duration_s,
        stim_amplitude=STIM_AMPLITUDE_CABLE,
        stim_index=1,
        store_history=True,
        history_stride=HISTORY_STRIDE,
    )

    v_matrix = result["v_matrix"]
    n_x = result["n_x"]
    dt_s = result["dt_s"]
    n_t = v_matrix.shape[0]

    # Account for history_stride in time array
    t_ms = np.arange(n_t) * dt_s * HISTORY_STRIDE * 1e3

    # Find index for recording position (midpoint)
    dx = L_um / n_x
    record_idx = int(record_x_um / dx)
    record_idx = min(record_idx, n_x - 1)

    # Find index for fast_model input (clean AP, away from stimulus artifact)
    source_idx = int(FAST_MODEL_INPUT_X_UM / dx)
    source_idx = min(source_idx, n_x - 1)

    V_record = v_matrix[:, record_idx]
    V_source = v_matrix[:, source_idx]  # Clean AP at x=1500μm for fast_model

    return {
        "t_ms": t_ms,
        "V_record": V_record,
        "V_source": V_source,
        "source_x_um": FAST_MODEL_INPUT_X_UM,
        "model": "hh_cable",
    }


def _run_hh_model(L_um: float, T_ms: float, record_x_um: float) -> dict:
    """
    Run hh_model simulation and extract V(t) at stimulus site.

    NOTE: hh_model has a diffusion coefficient bug that prevents AP propagation.
    We record at x=0 (stimulus site) where the AP is generated for shape comparison.

    hh_model uses cm natively, so we convert.

    Returns dict with:
        - t_ms: time array in ms
        - V_record: voltage at stimulus site (x=0)
        - V_source: voltage at stimulus site (x=0)
    """
    # Convert μm to cm
    L_cm = L_um * 1e-4
    T_s = T_ms * 1e-3
    stim_duration_s = STIM_DURATION_MS * 1e-3

    # Calculate n_spatial to match hh_cable resolution
    dx_cm = DX_UM * 1e-4  # Convert μm to cm
    n_spatial = max(10, int(L_cm / dx_cm))

    result = simulate_hh_model(
        length=L_cm,
        T_s=T_s,
        dt_s=DT_S,
        n_spatial=n_spatial,
        stim_start_s=0.0,
        stim_end_s=stim_duration_s,
        stim_amplitude=STIM_AMPLITUDE_MODEL,  # Use appropriate amplitude for hh_model
        stim_index=0,
        store_history=True,
        history_stride=HISTORY_STRIDE,
    )

    V = result["V"]
    t_s = result["t_s"]

    t_ms = t_s * 1e3

    # Record at stimulus site (x=0) since AP doesn't propagate in hh_model
    V_source = V[:, 0]

    return {
        "t_ms": t_ms,
        "V_record": V_source,  # Use x=0 since propagation is broken
        "V_source": V_source,
        "model": "hh_model",
    }


def _run_fast_model(
    v_source: np.ndarray,
    t_ms_source: np.ndarray,
    source_x_um: float,
    record_x_um: float,
) -> dict:
    """
    Run fast_model using HH waveform as input.
    Applies pure delay based on distance from source to recording position.

    Parameters
    ----------
    v_source : np.ndarray
        Input voltage waveform (from hh_cable at source_x_um)
    t_ms_source : np.ndarray
        Time array for input waveform
    source_x_um : float
        Position where input waveform was recorded
    record_x_um : float
        Position where we want to "record" the output

    Returns dict with:
        - t_ms: time array in ms
        - V_record: delayed voltage waveform
    """
    # Delay is the travel time from source position to recording position
    distance_um = record_x_um - source_x_um
    delay_ms = distance_um / CONDUCTION_VELOCITY

    result = simulate_fast_model(
        v_input=v_source,
        t_ms_input=t_ms_source,
        delay_ms=delay_ms,
        v_rest=-65.0,
    )

    return {
        "t_ms": result["t_ms"],
        "V_record": result["V"],
        "model": "fast_model",
        "delay_ms": delay_ms,
        "source_x_um": source_x_um,
        "record_x_um": record_x_um,
    }


# =============================================================================
# Alignment & Normalization
# =============================================================================


def _align_trace(
    t_ms: np.ndarray, V: np.ndarray, baseline_mV: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Align a single trace:
    1. Shift baseline to target value
    2. Shift time so peak is at t=0

    Returns (t_aligned, V_aligned, peak_amplitude)
    """
    # Baseline alignment: use resting potential (minimum in first half, avoiding AP undershoot)
    # This handles cases where stimulus starts at t=0
    half_len = len(V) // 2
    # Find resting potential as the most negative value before the peak
    peak_idx = np.argmax(V)
    pre_peak = V[: max(1, peak_idx)]
    # Use median of values below the 25th percentile as baseline estimate
    threshold = np.percentile(pre_peak, 25)
    baseline_mask = pre_peak <= threshold
    if np.any(baseline_mask):
        current_baseline = np.median(pre_peak[baseline_mask])
    else:
        current_baseline = np.min(pre_peak)

    V_shifted = V - current_baseline + baseline_mV

    # Peak alignment: find peak and shift time
    t_peak = t_ms[peak_idx]
    t_aligned = t_ms - t_peak

    peak_amplitude = V_shifted[peak_idx]

    return t_aligned, V_shifted, peak_amplitude


def _interpolate_to_grid(
    t_ms: np.ndarray, V: np.ndarray, t_ref: np.ndarray
) -> np.ndarray:
    """Interpolate trace to reference time grid."""
    return np.interp(t_ref, t_ms, V, left=np.nan, right=np.nan)


# =============================================================================
# Metrics Calculation
# =============================================================================


def _calculate_fwhm(t_ms: np.ndarray, V: np.ndarray, baseline_mV: float) -> float:
    """
    Calculate Full Width at Half Maximum.
    Assumes trace is already baseline-aligned.
    """
    peak_idx = np.argmax(V)
    peak_val = V[peak_idx]
    half_max = baseline_mV + (peak_val - baseline_mV) / 2

    # Find indices where V crosses half_max
    above_half = V > half_max
    if not np.any(above_half):
        return np.nan

    # Find first and last crossing
    crossings = np.where(np.diff(above_half.astype(int)))[0]
    if len(crossings) < 2:
        return np.nan

    t_start = t_ms[crossings[0]]
    t_end = t_ms[crossings[-1]]

    return t_end - t_start


def _calculate_rmse(
    V_test: np.ndarray, V_ref: np.ndarray, t_ref: np.ndarray, window_ms: float
) -> float:
    """
    Calculate RMSE between test and reference traces within ±window_ms of peak (t=0).
    Assumes traces are peak-aligned (peak at t=0).
    """
    # Select window around peak
    mask = (t_ref >= -window_ms) & (t_ref <= window_ms)
    mask &= ~np.isnan(V_test) & ~np.isnan(V_ref)

    if not np.any(mask):
        return np.nan

    V_test_win = V_test[mask]
    V_ref_win = V_ref[mask]

    rmse = np.sqrt(np.mean((V_test_win - V_ref_win) ** 2))
    return rmse


# =============================================================================
# Visualization
# =============================================================================


def _plot_comparison(traces: dict, output_path: str):
    """
    Create comparison plot with main view and inset zoom.

    traces: dict mapping model name -> (t_aligned, V_aligned)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot styling
    styles = {
        "hh_cable": {
            "color": "black",
            "linewidth": 2.5,
            "linestyle": "-",
            "label": "HH Cable @ midpoint (baseline)",
        },
        "hh_model": {
            "color": "blue",
            "linewidth": 1.5,
            "linestyle": "-",
            "label": "HH Model @ midpoint",
        },
        "fast_model": {
            "color": "red",
            "linewidth": 1.5,
            "linestyle": "--",
            "label": "Fast Model (delayed)",
        },
    }

    # Main plot
    for model_name, (t_ms, V) in traces.items():
        style = styles.get(
            model_name,
            {"color": "gray", "linewidth": 1, "linestyle": "-", "label": model_name},
        )
        ax.plot(t_ms, V, **style)

    ax.set_xlabel("Time relative to peak (ms)", fontsize=12)
    ax.set_ylabel("Membrane Potential (mV)", fontsize=12)
    ax.set_title("Action Potential Shape Comparison", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-15, 25)
    ax.axhline(BASELINE_MV, color="gray", linestyle=":", alpha=0.5, label="_baseline")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5, label="_peak")

    # Inset zoom [-2ms, +5ms]
    ax_inset = ax.inset_axes([0.55, 0.45, 0.4, 0.4])
    for model_name, (t_ms, V) in traces.items():
        style = styles.get(
            model_name, {"color": "gray", "linewidth": 1, "linestyle": "-"}
        )
        style_copy = {k: v for k, v in style.items() if k != "label"}
        ax_inset.plot(t_ms, V, **style_copy)

    ax_inset.set_xlim(-2, 5)
    ax_inset.set_title("Zoom: [-2ms, +5ms]", fontsize=9)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.axvline(0, color="gray", linestyle=":", alpha=0.5)

    # Mark the zoom region on main plot
    ax.indicate_inset_zoom(ax_inset, edgecolor="gray", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)  # Close instead of show to avoid blocking


def _print_metrics_table(metrics: list[dict]):
    """Print formatted metrics table to console."""
    print("\nAP Shape Comparison Results")
    print("=" * 60)
    print(f"{'Model':<15} {'FWHM (ms)':<12} {'Peak (mV)':<12} {'RMSE vs cable':<15}")
    print("-" * 60)

    for m in metrics:
        rmse_str = "---" if m["rmse"] is None else f"{m['rmse']:.4f}"
        fwhm_str = "N/A" if np.isnan(m["fwhm"]) else f"{m['fwhm']:.3f}"
        print(f"{m['model']:<15} {fwhm_str:<12} {m['peak']:<12.2f} {rmse_str:<15}")

    print("=" * 60)


# =============================================================================
# Main
# =============================================================================


def main():
    print("Running AP Shape Comparison Benchmark")
    print(f"  Axon length: {L_UM} μm")
    print(f"  Recording position: {RECORD_X_UM} μm (midpoint)")
    print(f"  Fast model input from: {FAST_MODEL_INPUT_X_UM} μm (clean AP)")
    print(f"  Simulation duration: {T_MS} ms")
    print(f"  Conduction velocity (fast_model): {CONDUCTION_VELOCITY} μm/ms")
    print()

    # Run simulations
    print("Running hh_cable simulation...")
    hh_cable_result = _run_hh_cable(L_UM, T_MS, RECORD_X_UM)

    print("Running hh_model simulation...")
    hh_model_result = _run_hh_model(L_UM, T_MS, RECORD_X_UM)

    print("Running fast_model simulation...")
    # Use hh_cable's clean AP waveform (at x=1500μm) as input to fast_model
    fast_model_result = _run_fast_model(
        v_source=hh_cable_result["V_source"],
        t_ms_source=hh_cable_result["t_ms"],
        source_x_um=hh_cable_result["source_x_um"],
        record_x_um=RECORD_X_UM,
    )
    print(f"  fast_model input from x={fast_model_result['source_x_um']} μm")
    print(f"  fast_model delay: {fast_model_result['delay_ms']:.2f} ms")

    # Align traces
    print("\nAligning traces...")
    results = {
        "hh_cable": hh_cable_result,
        "hh_model": hh_model_result,
        "fast_model": fast_model_result,
    }

    aligned_traces = {}
    peak_amplitudes = {}

    for name, res in results.items():
        t_aligned, V_aligned, peak_amp = _align_trace(
            res["t_ms"], res["V_record"], BASELINE_MV
        )
        aligned_traces[name] = (t_aligned, V_aligned)
        peak_amplitudes[name] = peak_amp

    # Interpolate all traces to hh_cable time grid for RMSE calculation
    t_ref = aligned_traces["hh_cable"][0]
    V_ref = aligned_traces["hh_cable"][1]

    interpolated = {"hh_cable": V_ref}
    for name in ["hh_model", "fast_model"]:
        t_aligned, V_aligned = aligned_traces[name]
        interpolated[name] = _interpolate_to_grid(t_aligned, V_aligned, t_ref)

    # Calculate metrics
    print("Calculating metrics...")
    metrics = []

    for name in ["hh_cable", "hh_model", "fast_model"]:
        t_aligned, V_aligned = aligned_traces[name]

        fwhm = _calculate_fwhm(t_aligned, V_aligned, BASELINE_MV)

        if name == "hh_cable":
            rmse = None
        else:
            rmse = _calculate_rmse(interpolated[name], V_ref, t_ref, RMSE_WINDOW_MS)

        metrics.append(
            {
                "model": name,
                "fwhm": fwhm,
                "peak": peak_amplitudes[name],
                "rmse": rmse,
            }
        )

    # Print metrics table
    _print_metrics_table(metrics)

    # Plot comparison
    _plot_comparison(aligned_traces, "ap_comparison_overlay.png")


if __name__ == "__main__":
    main()
