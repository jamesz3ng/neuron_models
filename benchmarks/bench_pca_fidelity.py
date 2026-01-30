"""
PCA Reconstruction Fidelity Benchmark

Measures how well the PCA-based encoding/decoding preserves action potential shape.

Metrics:
- RMSE: Root Mean Square Error (mV)
- Peak Error: Difference in peak amplitude (mV)
- FWHM Error: Difference in spike width at half-maximum (%)
- Rise Time Error: Difference in 10-90% rise time (%)
- AHP Error: Difference in after-hyperpolarization depth (mV)

Tests across different spike conditions:
- Normal spikes (single pulse, rested neuron)
- Fatigued spikes (end of high-frequency train)
- Population variation (±15% conductance scaling)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np

from src.simulation import DT_MS
from src.ssds_model import (
    _simulate_and_harvest,
    WINDOW_POINTS,
    WINDOW_PRE_MS,
    WINDOW_POST_MS,
    PRE_PEAK_POINTS,
)


# =============================================================================
# Shape Metrics
# =============================================================================


def measure_peak_amplitude(waveform: np.ndarray) -> float:
    """Peak voltage (mV)."""
    return float(np.max(waveform))


def measure_ahp_depth(waveform: np.ndarray, v_rest: float = -65.0) -> float:
    """After-hyperpolarization depth below rest (mV, positive = deeper)."""
    # Look in second half of waveform (after peak)
    post_peak = waveform[len(waveform) // 2 :]
    min_v = np.min(post_peak)
    return v_rest - min_v  # Positive if hyperpolarized below rest


def measure_fwhm(waveform: np.ndarray, dt_ms: float = DT_MS) -> float:
    """Full Width at Half Maximum (ms)."""
    peak_idx = np.argmax(waveform)
    peak_v = waveform[peak_idx]
    baseline = -65.0
    half_max = (peak_v + baseline) / 2.0

    # Find left crossing
    left_idx = peak_idx
    while left_idx > 0 and waveform[left_idx] > half_max:
        left_idx -= 1

    # Find right crossing
    right_idx = peak_idx
    while right_idx < len(waveform) - 1 and waveform[right_idx] > half_max:
        right_idx += 1

    return (right_idx - left_idx) * dt_ms


def measure_rise_time(waveform: np.ndarray, dt_ms: float = DT_MS) -> float:
    """10-90% rise time (ms)."""
    peak_idx = np.argmax(waveform)
    peak_v = waveform[peak_idx]
    baseline = waveform[0]  # Use start of window as baseline

    v_10 = baseline + 0.1 * (peak_v - baseline)
    v_90 = baseline + 0.9 * (peak_v - baseline)

    # Find crossings (searching backward from peak)
    idx_10 = peak_idx
    while idx_10 > 0 and waveform[idx_10] > v_10:
        idx_10 -= 1

    idx_90 = peak_idx
    while idx_90 > 0 and waveform[idx_90] > v_90:
        idx_90 -= 1

    return (idx_90 - idx_10) * dt_ms


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root Mean Square Error (mV)."""
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def compute_max_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Maximum absolute error (mV)."""
    return float(np.max(np.abs(original - reconstructed)))


# =============================================================================
# PCA Reconstruction
# =============================================================================


def load_pca_basis() -> tuple[np.ndarray, np.ndarray]:
    """Load mean waveform and principal components."""
    basis_path = _OUTPUT_DIR / "basis_data.npz"
    if not basis_path.exists():
        raise FileNotFoundError(f"{basis_path} not found. Run ssds_model.py first.")

    data = np.load(basis_path)
    return data["mean_waveform"], data["components"]


def reconstruct_spike(
    waveform: np.ndarray,
    mean_waveform: np.ndarray,
    components: np.ndarray,
    n_components: int = 3,
) -> np.ndarray:
    """Reconstruct spike using n principal components."""
    centered = waveform - mean_waveform
    weights = components[:n_components] @ centered
    return mean_waveform + weights @ components[:n_components]


# =============================================================================
# Test Spike Generation
# =============================================================================


def generate_normal_spikes(n_samples: int = 20) -> list[np.ndarray]:
    """Normal spikes from single pulses (rested neuron)."""
    print(f"  Generating {n_samples} normal spikes...")
    waveforms = []
    for i in range(n_samples):
        pulse_time = 5.0 + i * 50.0  # 50ms apart (fully rested)
        result = _simulate_and_harvest([pulse_time], extract_indices=[0])
        if result:
            waveforms.append(result[0]["waveform"])
    return waveforms


def generate_fatigued_spikes(n_trains: int = 10) -> list[np.ndarray]:
    """Fatigued spikes (last spike of 100Hz train)."""
    print(f"  Generating {n_trains} fatigued spikes (100Hz trains)...")
    waveforms = []
    for _ in range(n_trains):
        isi_ms = 10.0  # 100Hz
        pulse_times = [5.0 + i * isi_ms for i in range(15)]
        result = _simulate_and_harvest(pulse_times, extract_indices=[-1])
        if result:
            waveforms.append(result[0]["waveform"])
    return waveforms


def generate_population_spikes(n_samples: int = 50) -> list[np.ndarray]:
    """Spikes with ±15% conductance variation."""
    print(f"  Generating {n_samples} population variation spikes...")
    np.random.seed(42)
    waveforms = []
    for _ in range(n_samples):
        g_na_scale = np.random.uniform(0.85, 1.15)
        g_k_scale = np.random.uniform(0.85, 1.15)
        result = _simulate_and_harvest(
            [5.0],
            g_na_a_scale=g_na_scale,
            g_k_a_scale=g_k_scale,
            extract_indices=[0],
        )
        if result:
            waveforms.append(result[0]["waveform"])
    return waveforms


# =============================================================================
# Benchmark
# =============================================================================


def benchmark_reconstruction(
    waveforms: list[np.ndarray],
    mean_waveform: np.ndarray,
    components: np.ndarray,
    n_components: int = 3,
) -> dict:
    """Compute reconstruction metrics for a set of waveforms."""
    metrics = {
        "rmse": [],
        "max_error": [],
        "peak_error": [],
        "fwhm_error_pct": [],
        "rise_time_error_pct": [],
        "ahp_error": [],
    }

    for wf in waveforms:
        recon = reconstruct_spike(wf, mean_waveform, components, n_components)

        # Error metrics
        metrics["rmse"].append(compute_rmse(wf, recon))
        metrics["max_error"].append(compute_max_error(wf, recon))

        # Shape metrics
        orig_peak = measure_peak_amplitude(wf)
        recon_peak = measure_peak_amplitude(recon)
        metrics["peak_error"].append(abs(orig_peak - recon_peak))

        orig_fwhm = measure_fwhm(wf)
        recon_fwhm = measure_fwhm(recon)
        if orig_fwhm > 0:
            metrics["fwhm_error_pct"].append(
                100 * abs(orig_fwhm - recon_fwhm) / orig_fwhm
            )

        orig_rise = measure_rise_time(wf)
        recon_rise = measure_rise_time(recon)
        if orig_rise > 0:
            metrics["rise_time_error_pct"].append(
                100 * abs(orig_rise - recon_rise) / orig_rise
            )

        orig_ahp = measure_ahp_depth(wf)
        recon_ahp = measure_ahp_depth(recon)
        metrics["ahp_error"].append(abs(orig_ahp - recon_ahp))

    # Compute statistics
    stats = {}
    for key, values in metrics.items():
        if values:
            arr = np.array(values)
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "max": float(np.max(arr)),
            }

    return stats


def print_results_table(results: dict[str, dict]):
    """Print results as formatted table."""
    print("\n" + "=" * 90)
    print(
        f"{'Condition':<20} {'RMSE (mV)':<12} {'Peak Err':<12} {'FWHM Err %':<12} {'Rise Err %':<12} {'AHP Err':<12}"
    )
    print("-" * 90)

    for condition, stats in results.items():
        if not stats or "rmse" not in stats:
            print(
                f"{condition:<20} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}"
            )
            continue

        rmse = f"{stats['rmse']['mean']:.3f}±{stats['rmse']['std']:.3f}"
        peak = f"{stats['peak_error']['mean']:.3f}±{stats['peak_error']['std']:.3f}"
        fwhm = f"{stats['fwhm_error_pct']['mean']:.2f}±{stats['fwhm_error_pct']['std']:.2f}"
        rise = f"{stats['rise_time_error_pct']['mean']:.2f}±{stats['rise_time_error_pct']['std']:.2f}"
        ahp = f"{stats['ahp_error']['mean']:.3f}±{stats['ahp_error']['std']:.3f}"

        print(f"{condition:<20} {rmse:<12} {peak:<12} {fwhm:<12} {rise:<12} {ahp:<12}")

    print("=" * 90)


def plot_example_reconstructions(
    waveforms_by_condition: dict[str, list[np.ndarray]],
    mean_waveform: np.ndarray,
    components: np.ndarray,
    output_path: Path,
):
    """Plot example original vs reconstructed spikes for each condition."""
    import matplotlib.pyplot as plt

    # Use Inter font
    n_conditions = len(waveforms_by_condition)
    fig, axes = plt.subplots(
        1, n_conditions, figsize=(4 * n_conditions + 4, 4), sharey=True
    )

    if n_conditions == 1:
        axes = [axes]

    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    colors = {
        "Normal": "#55a868",
        "Fatigued": "#c44e52",
        "Population": "#8172b3",
    }

    for ax, (condition, waveforms) in zip(axes, waveforms_by_condition.items()):
        # Pick a representative spike
        wf = waveforms[len(waveforms) // 2]
        recon = reconstruct_spike(wf, mean_waveform, components, n_components=3)

        color = colors.get(condition, "#1a1a1a")

        ax.plot(t_ms, wf, color=color, linewidth=2, label="Original", alpha=0.9)
        ax.plot(
            t_ms,
            recon,
            color="#1a1a1a",
            linewidth=2,
            linestyle="--",
            label="Reconstructed",
        )

        rmse = compute_rmse(wf, recon)
        ax.set_title(f"{condition}\nRMSE = {rmse:.3f} mV", fontsize=12)
        ax.set_xlabel("Time (ms)", fontsize=10)
        ax.axhline(-65, color="#cccccc", linestyle=":", linewidth=1, alpha=0.5)
        ax.grid(False)

        if ax == axes[0]:
            ax.set_ylabel("Voltage (mV)", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("PCA RECONSTRUCTION FIDELITY BENCHMARK")
    print("=" * 70)

    # Load PCA basis
    print("\nLoading PCA basis...")
    mean_waveform, components = load_pca_basis()
    print(f"  Mean shape: {mean_waveform.shape}")
    print(f"  Components: {components.shape[0]} PCs")

    # Generate test spikes
    print("\nGenerating test spikes...")
    waveforms_by_condition = {
        "Normal": generate_normal_spikes(20),
        "Fatigued": generate_fatigued_spikes(10),
        "Population": generate_population_spikes(50),
    }

    # Benchmark each condition
    print("\nBenchmarking reconstruction fidelity...")
    results = {}
    for condition, waveforms in waveforms_by_condition.items():
        print(f"  {condition}: {len(waveforms)} spikes")
        results[condition] = benchmark_reconstruction(
            waveforms, mean_waveform, components, n_components=3
        )

    # Print results
    print_results_table(results)

    # Aggregate statistics
    all_waveforms = []
    for wfs in waveforms_by_condition.values():
        all_waveforms.extend(wfs)

    overall = benchmark_reconstruction(
        all_waveforms, mean_waveform, components, n_components=3
    )

    print("\n" + "-" * 70)
    print("OVERALL (all conditions combined)")
    print("-" * 70)
    print(
        f"  RMSE:          {overall['rmse']['mean']:.3f} ± {overall['rmse']['std']:.3f} mV"
    )
    print(
        f"  Peak Error:    {overall['peak_error']['mean']:.3f} ± {overall['peak_error']['std']:.3f} mV"
    )
    print(
        f"  FWHM Error:    {overall['fwhm_error_pct']['mean']:.2f} ± {overall['fwhm_error_pct']['std']:.2f} %"
    )
    print(
        f"  Rise Time Err: {overall['rise_time_error_pct']['mean']:.2f} ± {overall['rise_time_error_pct']['std']:.2f} %"
    )
    print(
        f"  AHP Error:     {overall['ahp_error']['mean']:.3f} ± {overall['ahp_error']['std']:.3f} mV"
    )
    print("-" * 70)

    # Plot examples
    plot_example_reconstructions(
        waveforms_by_condition,
        mean_waveform,
        components,
        _OUTPUT_DIR / "pca_fidelity_examples.png",
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
