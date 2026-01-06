"""
Debug HH Physics: Find Sweet Spot for Spike Frequency Adaptation

Sweeps stimulation frequencies to find the range where:
1. Spikes show visible adaptation (shorter/wider)
2. Neuron doesn't enter depolarization block

Goal: Identify optimal frequency for PCA library generation.
"""

import numpy as np
from math import exp as math_exp


# =============================================================================
# HH Model (copied from ssds_model.py for standalone execution)
# =============================================================================

DEFAULT_PARAMS = {
    "C_m": 1.0,
    "g_Na": 120.0,
    "g_K": 36.0,
    "g_L": 0.3,
    "E_Na": 50.0,
    "E_K": -77.0,
    "E_L": -54.387,
    "v_rest": -65.0,
}


def run_custom_hh(
    t_end_ms: float,
    dt_ms: float,
    i_stim_array: np.ndarray,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run HH model with custom stimulus current array."""
    C_m = DEFAULT_PARAMS["C_m"]
    g_Na = DEFAULT_PARAMS["g_Na"] * g_na_scale
    g_K = DEFAULT_PARAMS["g_K"] * g_k_scale
    g_L = DEFAULT_PARAMS["g_L"]
    E_Na = DEFAULT_PARAMS["E_Na"]
    E_K = DEFAULT_PARAMS["E_K"]
    E_L = DEFAULT_PARAMS["E_L"]
    v_rest = DEFAULT_PARAMS["v_rest"]

    inv_C_m = 1.0 / C_m
    n_time = int(t_end_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms

    if len(i_stim_array) != n_time:
        raise ValueError(
            f"i_stim_array length ({len(i_stim_array)}) != n_time ({n_time})"
        )

    V_val = v_rest
    m_val = 0.0529
    h_val = 0.5961
    n_val = 0.3177

    V_hist = np.zeros(n_time)
    V_hist[0] = V_val

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

        I_stim = i_stim_array[i - 1]
        dV = (I_stim - I_Na - I_K - I_L) * inv_C_m
        V_val += dV * dt_ms

        V_hist[i] = V_val

    return t_ms, V_hist


def create_pulse_train(
    t_end_ms: float,
    dt_ms: float,
    pulse_times_ms: list[float],
    pulse_duration_ms: float,
    amplitude: float,
) -> np.ndarray:
    """Create stimulus array with pulses at specified times."""
    n_time = int(t_end_ms / dt_ms)
    i_stim = np.zeros(n_time)

    for t_pulse in pulse_times_ms:
        start_idx = int(t_pulse / dt_ms)
        end_idx = int((t_pulse + pulse_duration_ms) / dt_ms)
        end_idx = min(end_idx, n_time)
        if start_idx < n_time:
            i_stim[start_idx:end_idx] = amplitude

    return i_stim


# =============================================================================
# Spike Analysis Functions
# =============================================================================

SPIKE_THRESHOLD_MV = -20.0


def find_spike_peaks(
    V: np.ndarray, threshold_mv: float = SPIKE_THRESHOLD_MV
) -> list[int]:
    """Find indices of spike peaks in voltage trace."""
    peaks = []
    in_spike = False
    max_v = -np.inf
    max_idx = 0

    for i in range(len(V)):
        if V[i] >= threshold_mv and not in_spike:
            in_spike = True
            max_v = V[i]
            max_idx = i
        elif in_spike and V[i] >= threshold_mv:
            if V[i] > max_v:
                max_v = V[i]
                max_idx = i
        elif in_spike and V[i] < threshold_mv:
            peaks.append(max_idx)
            in_spike = False
            max_v = -np.inf

    return peaks


def measure_spike_width(V: np.ndarray, peak_idx: int, dt_ms: float) -> float:
    """
    Measure spike width at half-maximum (FWHM).

    Returns width in ms, or -1 if measurement fails.
    """
    peak_v = V[peak_idx]
    baseline = -65.0  # Approximate resting potential
    half_max = (peak_v + baseline) / 2

    # Find left crossing
    left_idx = peak_idx
    while left_idx > 0 and V[left_idx] > half_max:
        left_idx -= 1

    # Find right crossing
    right_idx = peak_idx
    while right_idx < len(V) - 1 and V[right_idx] > half_max:
        right_idx += 1

    if left_idx == 0 or right_idx == len(V) - 1:
        return -1.0

    width_ms = (right_idx - left_idx) * dt_ms
    return width_ms


def extract_spike_window(
    V: np.ndarray,
    peak_idx: int,
    pre_points: int,
    post_points: int,
) -> np.ndarray | None:
    """Extract spike waveform aligned to peak."""
    start_idx = peak_idx - pre_points
    end_idx = peak_idx + post_points

    if start_idx < 0 or end_idx > len(V):
        return None

    return V[start_idx:end_idx].copy()


# =============================================================================
# Main Analysis
# =============================================================================


def run_frequency_sweep():
    """
    Sweep frequencies and analyze spike adaptation.
    """
    # Configuration
    dt_ms = 0.01
    stim_amplitude = 30.0
    stim_duration_ms = 0.5
    n_pulses = 20
    # Fine-grained sweep around the refractory boundary (~10ms = 100Hz)
    frequencies_hz = [50, 60, 70, 75, 80, 85, 90, 95, 100, 110, 120, 150, 200]

    # Window for spike extraction (2ms pre, 8ms post)
    pre_points = int(2.0 / dt_ms)
    post_points = int(8.0 / dt_ms)

    results = []
    traces = {}  # Store full traces for plotting
    spike_waveforms = {}  # Store 1st and last spike waveforms

    print("=" * 70)
    print("HH PHYSICS DEBUG: Spike Frequency Adaptation Analysis")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Stimulus amplitude: {stim_amplitude} nA/mm^2")
    print(f"  Pulse duration: {stim_duration_ms} ms")
    print(f"  Number of pulses: {n_pulses}")
    print(f"  Frequencies: {frequencies_hz} Hz")
    print()

    for freq_hz in frequencies_hz:
        isi_ms = 1000.0 / freq_hz

        # Create pulse train
        pulse_times = [5.0 + i * isi_ms for i in range(n_pulses)]
        t_end_ms = pulse_times[-1] + 20.0  # Extra time after last pulse

        i_stim = create_pulse_train(
            t_end_ms, dt_ms, pulse_times, stim_duration_ms, stim_amplitude
        )

        # Run simulation
        t_ms, V = run_custom_hh(t_end_ms, dt_ms, i_stim)

        # Store trace for key frequencies
        if freq_hz in [90, 100, 150]:
            traces[f"{freq_hz}Hz"] = (t_ms, V, i_stim)

        # Find spikes
        peaks = find_spike_peaks(V)
        n_spikes = len(peaks)

        # Analyze first and last spike
        if n_spikes >= 2:
            first_peak_idx = peaks[0]
            last_peak_idx = peaks[-1]

            first_height = V[first_peak_idx]
            last_height = V[last_peak_idx]
            height_change = ((last_height - first_height) / first_height) * 100

            first_width = measure_spike_width(V, first_peak_idx, dt_ms)
            last_width = measure_spike_width(V, last_peak_idx, dt_ms)
            width_change = (
                ((last_width - first_width) / first_width) * 100
                if first_width > 0
                else 0
            )

            # Extract waveforms for plotting
            first_waveform = extract_spike_window(
                V, first_peak_idx, pre_points, post_points
            )
            last_waveform = extract_spike_window(
                V, last_peak_idx, pre_points, post_points
            )

            if first_waveform is not None and last_waveform is not None:
                spike_waveforms[freq_hz] = {
                    "first": first_waveform,
                    "last": last_waveform,
                    "first_height": first_height,
                    "last_height": last_height,
                }

            results.append(
                {
                    "freq_hz": freq_hz,
                    "isi_ms": isi_ms,
                    "n_spikes": n_spikes,
                    "n_expected": n_pulses,
                    "first_height": first_height,
                    "last_height": last_height,
                    "height_change_pct": height_change,
                    "first_width": first_width,
                    "last_width": last_width,
                    "width_change_pct": width_change,
                }
            )
        else:
            results.append(
                {
                    "freq_hz": freq_hz,
                    "isi_ms": isi_ms,
                    "n_spikes": n_spikes,
                    "n_expected": n_pulses,
                    "first_height": V[peaks[0]] if peaks else np.nan,
                    "last_height": np.nan,
                    "height_change_pct": np.nan,
                    "first_width": np.nan,
                    "last_width": np.nan,
                    "width_change_pct": np.nan,
                }
            )

    return results, traces, spike_waveforms


def print_results_table(results: list[dict]):
    """Print formatted results table."""
    print("\n" + "=" * 90)
    print("RESULTS TABLE")
    print("=" * 90)
    print(
        f"{'Freq (Hz)':<10} {'ISI (ms)':<10} {'Spikes':<10} {'1st Height':<12} {'Last Height':<12} {'Height %':<10} {'Width %':<10}"
    )
    print("-" * 90)

    for r in results:
        spikes_str = f"{r['n_spikes']}/{r['n_expected']}"
        height_str = (
            f"{r['height_change_pct']:.1f}%"
            if not np.isnan(r["height_change_pct"])
            else "N/A"
        )
        width_str = (
            f"{r['width_change_pct']:.1f}%"
            if not np.isnan(r["width_change_pct"])
            else "N/A"
        )
        first_h = (
            f"{r['first_height']:.1f}" if not np.isnan(r["first_height"]) else "N/A"
        )
        last_h = f"{r['last_height']:.1f}" if not np.isnan(r["last_height"]) else "N/A"

        print(
            f"{r['freq_hz']:<10} {r['isi_ms']:<10.2f} {spikes_str:<10} {first_h:<12} {last_h:<12} {height_str:<10} {width_str:<10}"
        )

    print("=" * 90)

    # Interpretation
    print("\nINTERPRETATION:")
    for r in results:
        spike_ratio = r["n_spikes"] / r["n_expected"]
        if spike_ratio < 0.5:
            print(
                f"  {r['freq_hz']}Hz: SEVERE BLOCK - Only {r['n_spikes']}/{r['n_expected']} spikes ({spike_ratio * 100:.0f}%)"
            )
        elif spike_ratio < 0.9:
            print(
                f"  {r['freq_hz']}Hz: PARTIAL BLOCK - {r['n_spikes']}/{r['n_expected']} spikes, {r['height_change_pct']:.1f}% height"
            )
        elif not np.isnan(r["height_change_pct"]) and r["height_change_pct"] < -5:
            print(
                f"  {r['freq_hz']}Hz: GOOD ADAPTATION - {r['n_spikes']}/{r['n_expected']} spikes, {r['height_change_pct']:.1f}% height drop"
            )
        elif not np.isnan(r["height_change_pct"]):
            print(
                f"  {r['freq_hz']}Hz: MINIMAL ADAPTATION - {r['n_spikes']}/{r['n_expected']} spikes, {r['height_change_pct']:.1f}% height"
            )


def plot_results(
    results: list[dict],
    traces: dict,
    spike_waveforms: dict,
    output_path: str,
):
    """Create multi-panel diagnostic plot."""
    import matplotlib.pyplot as plt

    dt_ms = 0.01
    pre_ms = 2.0
    post_ms = 8.0
    window_points = int((pre_ms + post_ms) / dt_ms)
    t_spike = np.arange(window_points) * dt_ms - pre_ms

    # Layout: 2 rows
    # Top row: 5 panels for selected frequencies (1st vs last spike overlay)
    # Bottom row: 400Hz full trace + summary bar chart

    fig = plt.figure(figsize=(16, 10))

    # Top row: Spike overlays for selected frequencies
    display_freqs = [60, 75, 90, 100, 150]  # Select 5 key frequencies around sweet spot
    for i, freq in enumerate(display_freqs):
        ax = fig.add_subplot(2, 5, i + 1)

        if freq in spike_waveforms:
            wf = spike_waveforms[freq]
            ax.plot(
                t_spike,
                wf["first"],
                "b-",
                linewidth=2,
                label=f"1st ({wf['first_height']:.1f}mV)",
            )
            ax.plot(
                t_spike,
                wf["last"],
                "r--",
                linewidth=2,
                label=f"Last ({wf['last_height']:.1f}mV)",
            )

            # Find result for this frequency
            r = next((x for x in results if x["freq_hz"] == freq), None)
            if r and not np.isnan(r["height_change_pct"]):
                ax.set_title(
                    f"{freq}Hz\n{r['height_change_pct']:.1f}% height", fontsize=10
                )
            else:
                ax.set_title(f"{freq}Hz", fontsize=10)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{freq}Hz", fontsize=10)

        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(-20, color="gray", linestyle="--", alpha=0.3, label="Threshold")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("V (mV)")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-pre_ms, post_ms)

    # Bottom left: 90Hz full trace (near sweet spot)
    ax_trace = fig.add_subplot(2, 2, 3)

    # Store 90Hz trace for plotting
    trace_freq = 90
    if trace_freq not in traces:
        # Find closest available frequency
        available = list(traces.keys())
        if available:
            trace_freq = available[0]

    trace_key = (
        f"{trace_freq}Hz"
        if f"{trace_freq}Hz" in traces
        else list(traces.keys())[0]
        if traces
        else None
    )

    if trace_key and trace_key in traces:
        t_ms, V, i_stim = traces[trace_key]
        ax_trace.plot(t_ms, V, "b-", linewidth=0.8)
        ax_trace.axhline(
            -20, color="red", linestyle="--", alpha=0.5, label="Spike threshold"
        )
        ax_trace.axhline(
            0, color="orange", linestyle=":", alpha=0.5, label="Depol. block indicator"
        )
        ax_trace.set_xlabel("Time (ms)")
        ax_trace.set_ylabel("Voltage (mV)")
        ax_trace.set_title(
            f"{trace_key} Full Trace - Spike Train Analysis", fontsize=11
        )
        ax_trace.legend(loc="upper right", fontsize=8)
        ax_trace.grid(True, alpha=0.3)

        # Add stimulus indicator
        ax_stim = ax_trace.twinx()
        ax_stim.fill_between(
            t_ms,
            0,
            i_stim / max(i_stim) * 0.2,
            alpha=0.3,
            color="green",
            label="Stimulus",
        )
        ax_stim.set_ylim(0, 1)
        ax_stim.set_yticks([])

    # Bottom right: Summary bar chart
    ax_bar = fig.add_subplot(2, 2, 4)

    valid_results = [r for r in results if not np.isnan(r["height_change_pct"])]
    if valid_results:
        freqs_valid = [r["freq_hz"] for r in valid_results]
        height_changes = [r["height_change_pct"] for r in valid_results]
        width_changes = [r["width_change_pct"] for r in valid_results]

        x = np.arange(len(freqs_valid))
        width = 0.35

        bars1 = ax_bar.bar(
            x - width / 2,
            height_changes,
            width,
            label="Height change %",
            color="steelblue",
        )
        bars2 = ax_bar.bar(
            x + width / 2, width_changes, width, label="Width change %", color="coral"
        )

        ax_bar.axhline(0, color="black", linewidth=0.5)
        ax_bar.axhline(
            -10, color="green", linestyle="--", alpha=0.5, label="-10% threshold"
        )
        ax_bar.set_xlabel("Frequency (Hz)")
        ax_bar.set_ylabel("% Change (1st to Last spike)")
        ax_bar.set_title("Spike Frequency Adaptation Summary", fontsize=11)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(freqs_valid)
        ax_bar.legend(loc="lower left", fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars1, height_changes):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 2,
                f"{val:.1f}%",
                ha="center",
                va="top",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    plt.suptitle(
        "HH Model Stress Test: Finding Sweet Spot for Spike Frequency Adaptation",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)


def main():
    results, traces, spike_waveforms = run_frequency_sweep()
    print_results_table(results)
    plot_results(results, traces, spike_waveforms, "hh_stress_test.png")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find sweet spot: highest frequency with good spike survival and measurable adaptation
    # Criteria: >= 90% spike survival AND some adaptation (height drop > 2%)
    candidates = []
    for r in results:
        spike_ratio = r["n_spikes"] / r["n_expected"]
        if spike_ratio >= 0.9 and not np.isnan(r["height_change_pct"]):
            candidates.append(r)

    if candidates:
        # Sort by adaptation (most negative height change first)
        candidates.sort(key=lambda x: x["height_change_pct"])
        sweet_spot = candidates[0]

        print(f"  Sweet Spot Frequency: {sweet_spot['freq_hz']} Hz")
        print(f"  - ISI: {sweet_spot['isi_ms']:.2f} ms")
        print(
            f"  - Spikes: {sweet_spot['n_spikes']}/{sweet_spot['n_expected']} ({sweet_spot['n_spikes'] / sweet_spot['n_expected'] * 100:.0f}%)"
        )
        print(f"  - Height adaptation: {sweet_spot['height_change_pct']:.1f}%")
        print(f"  - Width adaptation: {sweet_spot['width_change_pct']:.1f}%")
        print(f"\n  Use this frequency for PCA library generation to capture")
        print(f"  meaningful spike shape variation without spike failure.")

        if len(candidates) > 1:
            print(f"\n  Other viable frequencies (>90% spike survival):")
            for c in candidates[1:4]:  # Show top 4
                print(
                    f"    {c['freq_hz']}Hz: {c['height_change_pct']:.1f}% height, {c['n_spikes']}/{c['n_expected']} spikes"
                )
    else:
        print("  No clear sweet spot found. Consider:")
        print("  - Increasing stimulus amplitude")
        print("  - Testing intermediate frequencies")
        print("  - Checking HH model parameters")

    print("=" * 70)


if __name__ == "__main__":
    main()
