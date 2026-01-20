"""
Spike Shape Decomposition System (SSDS)

Generates a library of diverse Action Potential shapes and performs PCA
to compress complex, history-dependent spike shapes into a small basis set
(Mean + 2-3 principal components).

Protocols:
- A (Fatigue Spectrum): 10-pulse trains at [75, 80, 85] Hz
- B (Refractory Curve): Paired pulses with ISI 2.0-20.0ms
- C (Population Heterogeneity): Single pulses with g_Na/g_K ±15% variation
- D (Hyperpolarization Rebound): Super-charged spikes from hyperpolarized states

Target: ~1,100 spikes to ensure robust PCA capturing full AP shape space.
"""

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .physics import HHPhysics
from .simulation import run_simulation, create_pulse_train, DT_MS, STIM_AMPLITUDE, STIM_DURATION_MS
from .analysis import find_spike_peaks, extract_aligned_spike

_OUTPUT_DIR = Path(__file__).parent.parent / "output"

# =============================================================================
# Configuration
# =============================================================================

# Spike extraction parameters
WINDOW_PRE_MS = 2.0  # ms before peak
WINDOW_POST_MS = 15  # ms after peak (increased for full AHP capture)
WINDOW_TOTAL_MS = WINDOW_PRE_MS + WINDOW_POST_MS
WINDOW_POINTS = int(WINDOW_TOTAL_MS / DT_MS)
PRE_PEAK_POINTS = int(WINDOW_PRE_MS / DT_MS)
POST_PEAK_POINTS = int(WINDOW_POST_MS / DT_MS)

SPIKE_THRESHOLD_MV = -20.0

# =============================================================================
# Harvester - Generic simulate-and-extract pipeline
# =============================================================================


def _simulate_and_harvest(
    pulse_times_ms: list[float],
    *,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
    v_init: float | None = None,
    gates_init: tuple[float, float, float] | None = None,
    metadata: dict | None = None,
    extract_indices: list[int] | None = None,
) -> list[dict]:
    """
    Run simulation and extract spike waveforms.

    Parameters
    ----------
    pulse_times_ms : list[float]
        Times of stimulus pulses.
    g_na_scale, g_k_scale : float
        Conductance scaling factors.
    v_init : float | None
        Initial voltage (uses steady-state gates if gates_init is None).
    gates_init : tuple[float, float, float] | None
        Initial (m, h, n) gate values.
    metadata : dict | None
        Additional metadata to attach to each spike.
    extract_indices : list[int] | None
        Which spike indices to extract (None = all).

    Returns
    -------
    list[dict]
        List of spike dictionaries with waveform and metadata.
    """
    # Simulation duration: last pulse + tail for AHP
    t_end_ms = pulse_times_ms[-1] + WINDOW_POST_MS + 5.0

    # Create stimulus and run simulation
    i_stim = create_pulse_train(t_end_ms, pulse_times_ms)
    t_ms, V = run_simulation(
        t_end_ms,
        i_stim,
        g_na_scale=g_na_scale,
        g_k_scale=g_k_scale,
        v_init=v_init,
        gates_init=gates_init,
    )

    # Find peaks
    peaks = find_spike_peaks(V)

    # Determine which spikes to extract
    if extract_indices is None:
        indices_to_extract = list(range(len(peaks)))
    else:
        indices_to_extract = [i for i in extract_indices if i < len(peaks)]

    # Extract waveforms
    spikes = []
    for i in indices_to_extract:
        peak_idx = peaks[i]
        waveform = extract_aligned_spike(V, peak_idx, PRE_PEAK_POINTS, POST_PEAK_POINTS)
        if waveform is not None:
            spike_data = {
                "waveform": waveform,
                "peak_idx": peak_idx,
                "peak_time_ms": peak_idx * DT_MS,
                "spike_num": i + 1,
            }
            if metadata:
                spike_data.update(metadata)
            spikes.append(spike_data)

    return spikes


# Protocol A: Fatigue Spectrum


def generate_protocol_a() -> list[dict]:
    # from input frequency test we can see that at 70hz there is ~0.2% diff between first and last spike -> indicates that the neuron is not tired/fatigued
    # 90hz ~ 7% diff
    # 100hz ~ 18% diff

    frequencies_hz = [60]
    print(f"Protocol A: Fatigue Spectrum (10 pulses at {frequencies_hz} Hz)...")

    all_spikes = []

    for freq_hz in frequencies_hz:
        isi_ms = 1000.0 / freq_hz
        pulse_times = [5.0 + i * isi_ms for i in range(10)]

        spikes = _simulate_and_harvest(
            pulse_times,
            metadata={"protocol": "A_fatigue", "freq_hz": freq_hz},
        )

        # Add frequency-specific spike numbering
        for s in spikes:
            s["freq_hz"] = freq_hz

        print(f"  {freq_hz}Hz: Found {len(spikes)} spikes")
        all_spikes.extend(spikes)

    print(f"  Total extracted: {len(all_spikes)} spikes")
    return all_spikes


# Protocol B: Refractory Curve


def generate_protocol_b() -> list[dict]:
    print("Protocol B: Refractory Curve (paired pulses, ISI 2.0-20.0ms, step 0.2ms)...")

    isi_values = np.arange(2.0, 20.2, 0.2)
    spikes = []

    for isi_ms in isi_values:
        pulse_times = [5.0, 5.0 + isi_ms]

        # Extract only the second spike (index 1)
        result = _simulate_and_harvest(
            pulse_times,
            metadata={"protocol": "B_refractory", "isi_ms": float(isi_ms)},
            extract_indices=[1],
        )
        spikes.extend(result)

    print(
        f"  Extracted {len(spikes)} second-spike waveforms (from {len(isi_values)} ISI values)"
    )
    return spikes


# Protocol C: Population Heterogeneity


def generate_protocol_c(n_samples: int = 1000, seed: int = 42) -> list[dict]:
    print(f"Protocol C: Population Heterogeneity ({n_samples} samples, ±15%)...")

    np.random.seed(seed)
    spikes = []

    for _ in range(n_samples):
        g_na_scale = np.random.uniform(0.5, 2)
        g_k_scale = np.random.uniform(0.5, 2)

        result = _simulate_and_harvest(
            [5.0],  # Single pulse
            g_na_scale=g_na_scale,
            g_k_scale=g_k_scale,
            metadata={
                "protocol": "C_population",
                "g_na_scale": g_na_scale,
                "g_k_scale": g_k_scale,
            },
            extract_indices=[0],
        )
        spikes.extend(result)

    print(f"  Extracted {len(spikes)} spikes")
    return spikes


# =============================================================================
# Protocol D: Hyperpolarization Rebound
# =============================================================================


def generate_protocol_d(n_samples: int = 50) -> list[dict]:
    print("Protocol D: Hyperpolarization Rebound (super-charged spikes)...")

    v_hyper_levels = [-70, -75, -80, -85, -90]
    samples_per_level = n_samples // len(v_hyper_levels)

    spikes = []

    for v_hyper in v_hyper_levels:
        # Compute steady-state gates at hyperpolarized voltage
        m_init, h_init, n_init = HHPhysics.steady_state(v_hyper)

        for _ in range(samples_per_level):
            result = _simulate_and_harvest(
                [5.0],  # Single pulse
                v_init=v_hyper,
                gates_init=(m_init, h_init, n_init),
                metadata={
                    "protocol": "D_hyperpol",
                    "v_hyper": v_hyper,
                    "h_init": h_init,
                },
                extract_indices=[0],
            )

            # Add peak voltage to metadata
            for s in result:
                s["peak_voltage"] = s["waveform"][PRE_PEAK_POINTS]

            spikes.extend(result)

        print(
            f"  V_hyper={v_hyper}mV: h_init={h_init:.3f}, extracted {samples_per_level} spikes"
        )

    print(f"  Total extracted: {len(spikes)} super-charged spikes")
    return spikes


# PCA Analysis (unchanged)


def run_pca_analysis(spikes: list[dict], n_components: int = 3) -> dict:
    """Run PCA on spike waveforms."""
    X = np.vstack([s["waveform"] for s in spikes])
    print(f"\nPCA Analysis: {X.shape[0]} spikes, {X.shape[1]} time points")

    pca = PCA(n_components=n_components)
    weights = pca.fit_transform(X)

    print(f"Explained Variance Ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {var * 100:.2f}%")
    print(
        f"  Total (first {n_components}): {sum(pca.explained_variance_ratio_) * 100:.2f}%"
    )

    return {
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
        "X": X,
        "weights": weights,
    }


def reconstruct_spike(
    pca_result: dict, spike_idx: int, n_components: int = 2
) -> np.ndarray:
    """Reconstruct a spike using mean + n_components PCs."""
    mean = pca_result["mean"]
    components = pca_result["components"]
    weights = pca_result["weights"][spike_idx]

    reconstructed = mean.copy()
    for i in range(n_components):
        reconstructed += weights[i] * components[i]

    return reconstructed


# Validation (unchanged logic, uses module constants)


def validate_reconstructions(spikes: list[dict], pca_result: dict) -> dict:
    """Comprehensive validation of reconstruction quality across spike types."""
    X = pca_result["X"]
    mean = pca_result["mean"]
    weights = pca_result["weights"]
    components = pca_result["components"]

    recon_2pc = mean + weights[:, :2] @ components[:2]
    errors_2pc = np.sqrt(np.mean((X - recon_2pc) ** 2, axis=1))

    recon_3pc = mean + weights[:, :3] @ components[:3]
    errors_3pc = np.sqrt(np.mean((X - recon_3pc) ** 2, axis=1))

    validation = {"test_cases": [], "summary": {}}

    def add_case(name: str, idx: int, description: str):
        spike = spikes[idx]
        validation["test_cases"].append(
            {
                "name": name,
                "description": description,
                "idx": idx,
                "protocol": spike["protocol"],
                "rmse_2pc": errors_2pc[idx],
                "rmse_3pc": errors_3pc[idx],
                "weights": weights[idx].tolist(),
                "spike_meta": {k: v for k, v in spike.items() if k != "waveform"},
            }
        )

    # Protocol A: Fatigue progression
    fatigue_indices = [i for i, s in enumerate(spikes) if s["protocol"] == "A_fatigue"]
    if fatigue_indices:
        freq_groups = {}
        for idx in fatigue_indices:
            freq = spikes[idx].get("freq_hz", 0)
            freq_groups.setdefault(freq, []).append(idx)

        for freq in sorted(freq_groups.keys()):
            indices = freq_groups[freq]
            if indices:
                add_case(
                    f"A_fresh_{freq}Hz",
                    indices[0],
                    f"First spike at {freq}Hz (fresh neuron)",
                )
            if len(indices) >= 2:
                add_case(
                    f"A_fatigued_{freq}Hz",
                    indices[-1],
                    f"Last spike at {freq}Hz (fatigued)",
                )

    # Protocol B: Refractory recovery
    refractory_indices = [
        i for i, s in enumerate(spikes) if s["protocol"] == "B_refractory"
    ]
    if len(refractory_indices) >= 3:
        sorted_refr = sorted(
            refractory_indices, key=lambda i: spikes[i].get("isi_ms", 0)
        )
        idx = sorted_refr[0]
        add_case(
            "B_short_ISI",
            idx,
            f"Short ISI={spikes[idx]['isi_ms']:.1f}ms (partial recovery)",
        )
        mid_idx = sorted_refr[len(sorted_refr) // 2]
        add_case(
            "B_medium_ISI", mid_idx, f"Medium ISI={spikes[mid_idx]['isi_ms']:.1f}ms"
        )
        idx = sorted_refr[-1]
        add_case(
            "B_long_ISI", idx, f"Long ISI={spikes[idx]['isi_ms']:.1f}ms (full recovery)"
        )

    # Protocol C: Population extremes
    pop_indices = [i for i, s in enumerate(spikes) if s["protocol"] == "C_population"]
    if pop_indices:
        high_na_low_k = max(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            - spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_high_gNa_low_gK",
            high_na_low_k,
            f"High gNa={spikes[high_na_low_k]['g_na_scale']:.2f}, Low gK={spikes[high_na_low_k]['g_k_scale']:.2f}",
        )

        low_na_high_k = min(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            - spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_low_gNa_high_gK",
            low_na_high_k,
            f"Low gNa={spikes[low_na_high_k]['g_na_scale']:.2f}, High gK={spikes[low_na_high_k]['g_k_scale']:.2f}",
        )

        both_high = max(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            + spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_both_high",
            both_high,
            f"Both high: gNa={spikes[both_high]['g_na_scale']:.2f}, gK={spikes[both_high]['g_k_scale']:.2f}",
        )

        both_low = min(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            + spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_both_low",
            both_low,
            f"Both low: gNa={spikes[both_low]['g_na_scale']:.2f}, gK={spikes[both_low]['g_k_scale']:.2f}",
        )

    # Protocol D: Hyperpolarization rebound
    hyperpol_indices = [
        i for i, s in enumerate(spikes) if s["protocol"] == "D_hyperpol"
    ]
    if hyperpol_indices:
        hyper_groups = {}
        for idx in hyperpol_indices:
            v_hyper = spikes[idx].get("v_hyper", -65)
            hyper_groups.setdefault(v_hyper, []).append(idx)

        most_hyper = min(hyper_groups.keys())
        if hyper_groups[most_hyper]:
            idx = hyper_groups[most_hyper][0]
            peak_v = spikes[idx].get("peak_voltage", 0)
            add_case(
                "D_super_charged",
                idx,
                f"V_hyper={most_hyper}mV, h={spikes[idx]['h_init']:.2f}, peak={peak_v:.1f}mV",
            )

        least_hyper = max(hyper_groups.keys())
        if hyper_groups[least_hyper] and least_hyper != most_hyper:
            idx = hyper_groups[least_hyper][0]
            peak_v = spikes[idx].get("peak_voltage", 0)
            add_case(
                "D_mild_hyper",
                idx,
                f"V_hyper={least_hyper}mV, h={spikes[idx]['h_init']:.2f}, peak={peak_v:.1f}mV",
            )

    # Worst and best cases
    worst_idx = int(np.argmax(errors_2pc))
    add_case(
        "Worst_case",
        worst_idx,
        f"Highest reconstruction error (RMSE={errors_2pc[worst_idx]:.2f}mV)",
    )

    best_idx = int(np.argmin(errors_2pc))
    add_case(
        "Best_case",
        best_idx,
        f"Lowest reconstruction error (RMSE={errors_2pc[best_idx]:.2f}mV)",
    )

    # Summary statistics
    validation["summary"] = {
        "n_test_cases": len(validation["test_cases"]),
        "mean_rmse_2pc": float(
            np.mean([tc["rmse_2pc"] for tc in validation["test_cases"]])
        ),
        "max_rmse_2pc": float(
            np.max([tc["rmse_2pc"] for tc in validation["test_cases"]])
        ),
        "mean_rmse_3pc": float(
            np.mean([tc["rmse_3pc"] for tc in validation["test_cases"]])
        ),
        "all_spikes_mean_rmse_2pc": float(np.mean(errors_2pc)),
        "all_spikes_max_rmse_2pc": float(np.max(errors_2pc)),
        "all_spikes_percentile_95_rmse_2pc": float(np.percentile(errors_2pc, 95)),
        "all_spikes_percentile_99_rmse_2pc": float(np.percentile(errors_2pc, 99)),
        "max_rmse_3pc": float(
            np.max([tc["rmse_3pc"] for tc in validation["test_cases"]])
        ),
        "all_spikes_mean_rmse_3pc": float(np.mean(errors_3pc)),
        "all_spikes_max_rmse_3pc": float(np.max(errors_3pc)),
        "all_spikes_percentile_95_rmse_3pc": float(np.percentile(errors_3pc, 95)),
        "all_spikes_percentile_99_rmse_3pc": float(np.percentile(errors_3pc, 99)),
    }

    return validation


# =============================================================================
# Visualization (unchanged, uses module constants)
# =============================================================================


def plot_validation(
    spikes: list[dict], pca_result: dict, validation: dict, output_path: str
):
    """Create multi-panel validation figure."""
    import matplotlib.pyplot as plt

    test_cases = validation["test_cases"]
    n_cases = len(test_cases)

    n_cols = 3
    n_rows = max((n_cases + n_cols - 1) // n_cols, 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    for i, tc in enumerate(test_cases):
        ax = axes[i]
        idx = tc["idx"]

        actual = spikes[idx]["waveform"]
        recon_2pc = reconstruct_spike(pca_result, idx, n_components=2)
        recon_3pc = reconstruct_spike(pca_result, idx, n_components=3)

        ax.plot(t_ms, actual, "k-", linewidth=2, label="Actual")
        ax.plot(
            t_ms,
            pca_result["mean"],
            "gray",
            linewidth=1,
            linestyle=":",
            alpha=0.5,
            label="Mean",
        )
        ax.plot(
            t_ms,
            recon_2pc,
            "r--",
            linewidth=1.5,
            label=f"2PC (RMSE={tc['rmse_2pc']:.2f})",
        )
        ax.plot(
            t_ms,
            recon_3pc,
            "b:",
            linewidth=1.5,
            label=f"3PC (RMSE={tc['rmse_3pc']:.2f})",
        )

        ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"{tc['name']}\n{tc['description']}", fontsize=9)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("V (mV)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

        if tc["rmse_2pc"] < 0.3:
            ax.set_facecolor("#e6ffe6")
        elif tc["rmse_2pc"] < 0.5:
            ax.set_facecolor("#ffffcc")
        else:
            ax.set_facecolor("#ffe6e6")

    for i in range(n_cases, len(axes)):
        axes[i].axis("off")

    if n_cases < len(axes):
        ax_summary = axes[n_cases]
        ax_summary.axis("off")
        summary = validation["summary"]
        summary_text = (
            f"VALIDATION SUMMARY\n{'=' * 30}\n\n"
            f"Test Cases: {summary['n_test_cases']}\n\n"
            f"Selected Test Cases:\n"
            f"  Mean RMSE (2PC): {summary['mean_rmse_2pc']:.3f} mV\n"
            f"  Max RMSE (2PC):  {summary['max_rmse_2pc']:.3f} mV\n"
            f"  Mean RMSE (3PC): {summary['mean_rmse_3pc']:.3f} mV\n"
            f"  Max RMSE (3PC):  {summary['max_rmse_3pc']:.3f} mV\n\n"
            f"All {len(spikes)} Spikes:\n"
            f"  Mean RMSE (2PC): {summary['all_spikes_mean_rmse_2pc']:.3f} mV\n"
            f"  Max RMSE (2PC):  {summary['all_spikes_max_rmse_2pc']:.3f} mV\n"
            f"  95th %ile:       {summary['all_spikes_percentile_95_rmse_2pc']:.3f} mV\n"
            f"  99th %ile:       {summary['all_spikes_percentile_99_rmse_2pc']:.3f} mV\n\n"
            f"  Mean RMSE (3PC): {summary['all_spikes_mean_rmse_3pc']:.3f} mV\n"
            f"  Max RMSE (3PC):  {summary['all_spikes_max_rmse_3pc']:.3f} mV\n"
            f"  95th %ile:       {summary['all_spikes_percentile_95_rmse_3pc']:.3f} mV\n"
            f"  99th %ile:       {summary['all_spikes_percentile_99_rmse_3pc']:.3f} mV"
        )
        ax_summary.text(
            0.1,
            0.9,
            summary_text,
            transform=ax_summary.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

    plt.suptitle(
        "Basis Function Validation: Reconstruction Quality Across Spike Types",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=150)
    print(f"Saved validation plot to: {output_path}")
    plt.close(fig)


def plot_results(spikes: list[dict], pca_result: dict, output_path: str):
    """Create 3-panel figure showing spike library, eigen-spikes, and reconstruction."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    # Panel 1: All spikes + Mean
    ax1 = axes[0]
    colors = {
        "A_fatigue": "red",
        "B_refractory": "blue",
        "C_population": "gray",
        "D_hyperpol": "green",
    }

    for s in spikes:
        color = colors.get(s["protocol"], "gray")
        alpha = 0.3 if s["protocol"] == "C_population" else 0.5
        ax1.plot(t_ms, s["waveform"], color=color, alpha=alpha, linewidth=0.5)

    ax1.plot(t_ms, pca_result["mean"], "k-", linewidth=2.5, label="Mean")

    for proto, color in colors.items():
        count = sum(1 for s in spikes if s["protocol"] == proto)
        ax1.plot(
            [],
            [],
            color=color,
            linewidth=2,
            label=f"{proto.split('_')[1].capitalize()} (n={count})",
        )

    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.axhline(-65, color="gray", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Time relative to peak (ms)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title(f"Spike Library: {len(spikes)} aligned spikes")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    # Panel 2:
    ax2 = axes[1]
    components = pca_result["components"]
    var_ratios = pca_result["explained_variance_ratio"]
    scale = 20.0

    ax2.plot(
        t_ms,
        components[0] * scale,
        "r-",
        linewidth=2,
        label=f"PC1 ({var_ratios[0] * 100:.1f}%)",
    )
    ax2.plot(
        t_ms,
        components[1] * scale,
        "b-",
        linewidth=2,
        label=f"PC2 ({var_ratios[1] * 100:.1f}%)",
    )
    if len(components) > 2:
        ax2.plot(
            t_ms,
            components[2] * scale,
            "g-",
            linewidth=2,
            label=f"PC3 ({var_ratios[2] * 100:.1f}%)",
        )

    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Time relative to peak (ms)")
    ax2.set_ylabel(f"PC amplitude (scaled by {scale})")
    ax2.set_title("Principal Components")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    # Panel 3: Reconstruction example
    ax3 = axes[2]
    fatigue_spikes = [i for i, s in enumerate(spikes) if s["protocol"] == "A_fatigue"]

    if fatigue_spikes:
        target_idx = fatigue_spikes[-1]
        target_spike = spikes[target_idx]

        actual = target_spike["waveform"]
        recon_2pc = reconstruct_spike(pca_result, target_idx, n_components=2)
        recon_3pc = reconstruct_spike(pca_result, target_idx, n_components=3)

        rmse_2pc = np.sqrt(np.mean((actual - recon_2pc) ** 2))
        rmse_3pc = np.sqrt(np.mean((actual - recon_3pc) ** 2))

        ax3.plot(t_ms, actual, "k-", linewidth=2.5, label="Actual (Most Fatigued)")
        ax3.plot(
            t_ms,
            pca_result["mean"],
            "gray",
            linewidth=1.5,
            linestyle="--",
            label="Mean only",
        )
        ax3.plot(
            t_ms,
            recon_2pc,
            "r--",
            linewidth=2,
            label=f"Mean + 2 PCs (RMSE={rmse_2pc:.2f} mV)",
        )
        ax3.plot(
            t_ms,
            recon_3pc,
            "b:",
            linewidth=2,
            label=f"Mean + 3 PCs (RMSE={rmse_3pc:.2f} mV)",
        )

        weights = pca_result["weights"][target_idx]
        weight_text = f"Weights: w1={weights[0]:.1f}, w2={weights[1]:.1f}"
        if len(weights) > 2:
            weight_text += f", w3={weights[2]:.1f}"
        ax3.text(
            0.02,
            0.98,
            weight_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        title = f"Reconstruction: Spike #{target_spike['spike_num']} from Fatigue Train"
    else:
        actual = spikes[0]["waveform"]
        recon_2pc = reconstruct_spike(pca_result, 0, n_components=2)
        ax3.plot(t_ms, actual, "k-", linewidth=2.5, label="Actual")
        ax3.plot(t_ms, recon_2pc, "r--", linewidth=2, label="Mean + 2 PCs")
        title = "Reconstruction: First Spike"

    ax3.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Time relative to peak (ms)")
    ax3.set_ylabel("Voltage (mV)")
    ax3.set_title(title)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("SPIKE SHAPE DECOMPOSITION SYSTEM (SSDS)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Time step: {DT_MS} ms")
    print(
        f"  Window: -{WINDOW_PRE_MS}ms to +{WINDOW_POST_MS}ms ({WINDOW_POINTS} points)"
    )
    print(f"  Spike threshold: {SPIKE_THRESHOLD_MV} mV")
    print()

    # Generate spike library
    print("-" * 70)
    print("GENERATING SPIKE LIBRARY")
    print("-" * 70)

    spikes_a = generate_protocol_a()
    spikes_b = generate_protocol_b()
    spikes_c = generate_protocol_c(n_samples=1000)
    spikes_d = generate_protocol_d(n_samples=50)

    all_spikes = spikes_a + spikes_b + spikes_c + spikes_d
    print(f"\nTotal spikes collected: {len(all_spikes)}")

    # Run PCA
    print("-" * 70)
    print("PCA ANALYSIS")
    print("-" * 70)

    pca_result = run_pca_analysis(all_spikes, n_components=3)

    # Compute reconstruction errors
    print("\nReconstruction Error Analysis:")
    X = pca_result["X"]
    mean = pca_result["mean"]

    mean_only_error = np.sqrt(np.mean((X - mean) ** 2, axis=1))
    print(
        f"  Mean only:  RMSE = {np.mean(mean_only_error):.2f} ± {np.std(mean_only_error):.2f} mV"
    )

    recon_2pc = mean + pca_result["weights"][:, :2] @ pca_result["components"][:2]
    error_2pc = np.sqrt(np.mean((X - recon_2pc) ** 2, axis=1))
    print(f"  Mean + 2PC: RMSE = {np.mean(error_2pc):.2f} ± {np.std(error_2pc):.2f} mV")

    recon_3pc = mean + pca_result["weights"][:, :3] @ pca_result["components"][:3]
    error_3pc = np.sqrt(np.mean((X - recon_3pc) ** 2, axis=1))
    print(f"  Mean + 3PC: RMSE = {np.mean(error_3pc):.2f} ± {np.std(error_3pc):.2f} mV")

    # Visualize
    print("-" * 70)
    print("VISUALIZATION")
    print("-" * 70)

    plot_results(all_spikes, pca_result, _OUTPUT_DIR / "spike_pca_analysis.png")

    # Validation
    print("-" * 70)
    print("RECONSTRUCTION VALIDATION")
    print("-" * 70)

    validation = validate_reconstructions(all_spikes, pca_result)

    print(f"\nValidation Test Cases ({validation['summary']['n_test_cases']} spikes):")
    for tc in validation["test_cases"]:
        status = "OK" if tc["rmse_2pc"] < 0.5 else "REVIEW"
        print(
            f"  [{status}] {tc['name']}: RMSE={tc['rmse_2pc']:.3f}mV - {tc['description']}"
        )

    print(f"\nValidation Summary:")
    summary = validation["summary"]
    print(f"  Test cases mean RMSE (2PC): {summary['mean_rmse_2pc']:.3f} mV")
    print(f"  Test cases max RMSE (2PC):  {summary['max_rmse_2pc']:.3f} mV")
    print(
        f"  All spikes 95th percentile: {summary['all_spikes_percentile_95_rmse_2pc']:.3f} mV"
    )
    print(
        f"  All spikes 99th percentile: {summary['all_spikes_percentile_99_rmse_2pc']:.3f} mV"
    )

    plot_validation(all_spikes, pca_result, validation, _OUTPUT_DIR / "spike_validation.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_var = sum(pca_result["explained_variance_ratio"][:2]) * 100
    print(f"  2 PCs capture {total_var:.1f}% of variance")
    print(f"  Mean reconstruction error with 2 PCs: {np.mean(error_2pc):.2f} mV")
    print(f"  This proves we can represent complex spike shapes with:")
    print(f"    - 1 static Mean waveform ({WINDOW_POINTS} floats)")
    print(f"    - 2 PC basis vectors ({2 * WINDOW_POINTS} floats)")
    print(f"    - 2 dynamic weights per spike (2 floats)")
    print("=" * 70)

    # Export Basis Functions
    print("\n" + "-" * 70)
    print("EXPORTING BASIS FUNCTIONS")
    print("-" * 70)

    components_3 = pca_result["components"][:3]
    explained_variance_3 = pca_result["explained_variance_ratio"][:3]

    output_file = _OUTPUT_DIR / "basis_data.npz"
    np.savez(
        output_file,
        mean_waveform=pca_result["mean"],
        components=components_3,
        explained_variance=explained_variance_3,
        dt_ms=DT_MS,
        window_pre_ms=WINDOW_PRE_MS,
        window_post_ms=WINDOW_POST_MS,
        window_samples=WINDOW_POINTS,
    )

    print(f"  Saved basis functions to: {output_file}")
    print(f"  Contents:")
    print(f"    - mean_waveform: shape {pca_result['mean'].shape}")
    print(f"    - components: shape {components_3.shape}")
    print(f"    - explained_variance: {explained_variance_3}")
    print(f"    - dt_ms: {DT_MS}")
    print(f"    - window_pre_ms: {WINDOW_PRE_MS}")
    print(f"    - window_post_ms: {WINDOW_POST_MS}")
    print(f"    - window_samples: {WINDOW_POINTS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
