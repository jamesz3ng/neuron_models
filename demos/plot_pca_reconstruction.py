"""
Poster-ready figure demonstrating PCA reconstruction accuracy.

Shows original vs reconstructed action potentials across different conditions,
with clean overlay comparison and key metrics.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np
import matplotlib.pyplot as plt

# Use Inter font
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2

from src.simulation import DT_MS
from src.ssds_model import (
    _simulate_and_harvest,
    WINDOW_POINTS,
    WINDOW_PRE_MS,
    WINDOW_POST_MS,
)

# Seaborn deep palette
COLORS = {
    "original": "#1a1a1a",  # Near black
    "reconstructed": "#c44e52",  # Deep red
}


def load_pca_basis() -> tuple[np.ndarray, np.ndarray]:
    """Load mean waveform and principal components."""
    basis_path = _OUTPUT_DIR / "basis_data.npz"
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


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root Mean Square Error (mV)."""
    return float(np.sqrt(np.mean((original - reconstructed) ** 2)))


def generate_example_spikes() -> dict[str, np.ndarray]:
    """Generate one representative spike for each condition."""
    spikes = {}

    # Normal spike (single pulse, rested)
    print("  Generating normal spike...")
    result = _simulate_and_harvest([5.0], extract_indices=[0])
    if result:
        spikes["Normal"] = result[0]["waveform"]

    # Fatigued spike (last of 100Hz train)
    print("  Generating fatigued spike...")
    isi_ms = 10.0  # 100Hz
    pulse_times = [5.0 + i * isi_ms for i in range(15)]
    result = _simulate_and_harvest(pulse_times, extract_indices=[-1])
    if result:
        spikes["Fatigued"] = result[0]["waveform"]

    # Population variation (altered conductances)
    print("  Generating population variation spike...")
    result = _simulate_and_harvest(
        [5.0],
        g_na_a_scale=0.9,
        g_k_a_scale=1.1,
        extract_indices=[0],
    )
    if result:
        spikes["Population"] = result[0]["waveform"]

    return spikes


def main():
    print("=" * 60)
    print("POSTER FIGURE: PCA Reconstruction Accuracy")
    print("=" * 60)

    # Load basis
    print("\nLoading PCA basis...")
    mean_waveform, components = load_pca_basis()

    # Generate spikes
    print("\nGenerating example spikes...")
    spikes = generate_example_spikes()

    # Time axis
    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    # Create figure - 1x2 horizontal layout (Fatigued, Population)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    conditions = ["Fatigued", "Population"]
    subtitles = [
        "Dynamics after 100 Hz train",
        "Conductance variations in Na and K channels",
    ]

    rmse_values = []

    for ax, condition, subtitle in zip(axes, conditions, subtitles):
        if condition not in spikes:
            ax.set_visible(False)
            continue

        wf = spikes[condition]
        recon = reconstruct_spike(wf, mean_waveform, components, n_components=3)
        rmse = compute_rmse(wf, recon)
        rmse_values.append((condition, rmse))

        # Plot original
        ax.plot(
            t_ms,
            wf,
            color=COLORS["original"],
            linewidth=2.5,
            label="Original",
            zorder=2,
        )

        # Plot reconstructed (dashed)
        ax.plot(
            t_ms,
            recon,
            color=COLORS["reconstructed"],
            linewidth=2.5,
            linestyle="--",
            label="Reconstructed",
            zorder=3,
        )

        # Styling
        ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)
        ax.set_ylim(-85, 50)

        # Reference lines
        ax.axhline(-65, color="#cccccc", linestyle=":", linewidth=1, alpha=0.6)
        ax.axvline(0, color="#cccccc", linestyle=":", linewidth=1, alpha=0.6)

        # Title with condition and RMSE
        ax.set_title(
            f"{condition}\n{subtitle}",
            fontsize=14,
            fontweight="bold",
            pad=12,
        )

        # RMSE annotation box
        ax.annotate(
            f"RMSE = {rmse:.2f} mV",
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=COLORS["reconstructed"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=COLORS["reconstructed"],
                linewidth=1.5,
                alpha=0.9,
            ),
        )

        ax.set_xlabel("Time relative to peak (ms)", fontsize=12)
        ax.set_ylabel("Voltage (mV)", fontsize=12)
        ax.tick_params(labelsize=11)

        # Minimal grid
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)



    # Add legend to first subplot only
    axes[0].legend(
        loc="upper right",
        fontsize=11,
        framealpha=0.95,
        edgecolor="none",
    )

    # Overall title
    fig.suptitle(
        "PCA Reconstruction Fidelity (3 Components)",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    # Save
    output_path = _OUTPUT_DIR / "poster_pca_reconstruction.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved: {output_path}")
    print("\nRMSE Summary:")
    for condition, rmse in rmse_values:
        print(f"  {condition}: {rmse:.3f} mV")

    avg_rmse = np.mean([r for _, r in rmse_values])
    print(f"\n  Average: {avg_rmse:.3f} mV")
    print("=" * 60)


if __name__ == "__main__":
    main()
