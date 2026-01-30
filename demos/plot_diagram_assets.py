"""
Generate clean, high-resolution "sticker" images for flowchart diagrams.

Assets:
1. icon_spike_cloud.png - Messy biological reality (200 gray traces + mean)
2. icon_components.png - The mathematical ingredients (PC1, PC2, PC3)
3. icon_reconstruction.png - It fits perfectly (actual vs reconstructed)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np
import matplotlib.pyplot as plt

# Use Inter font for all text
plt.rcParams["font.family"] = "Inter"

from src.ssds_model import (
    _simulate_and_harvest,
    WINDOW_POINTS,
    WINDOW_PRE_MS,
    WINDOW_POST_MS,
    DT_MS,
    PRE_PEAK_POINTS,
)

# Seaborn deep palette
COLORS = {
    "pc1": "#c44e52",  # deep red
    "pc2": "#4c72b0",  # deep blue
    "pc3": "#55a868",  # deep green
    "mean": "#1a1a1a",  # near black
    "cloud": "#8c8c8c",  # neutral gray
    "recon": "#c44e52",  # deep red for reconstruction
}


def _setup_schematic_axes(ax):
    """Configure axes for clean schematic look."""
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_spike_cloud(n_samples: int = 200, seed: int = 42) -> list[np.ndarray]:
    """Generate population heterogeneity spikes (Protocol C subset)."""
    print(f"  Generating {n_samples} population spikes...")
    np.random.seed(seed)
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


def generate_fatigued_spike() -> tuple[np.ndarray, dict]:
    """Generate a fatigued spike from Protocol A (high frequency train)."""
    print("  Generating fatigued spike (100Hz train)...")
    freq_hz = 100
    isi_ms = 1000.0 / freq_hz
    pulse_times = [5.0 + i * isi_ms for i in range(20)]

    spikes = _simulate_and_harvest(
        pulse_times,
        metadata={"protocol": "A_fatigue", "freq_hz": freq_hz},
    )

    # Return the last (most fatigued) spike
    if spikes:
        return spikes[-1]["waveform"], spikes[-1]
    return None, None


def plot_spike_cloud(
    waveforms: list[np.ndarray],
    output_path: Path,
    *,
    color: str = None,
):
    """Asset 1: Spike cloud (no mean overlay)."""
    print(f"  Creating spike cloud asset...")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    _setup_schematic_axes(ax)

    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    # Reference line at rest
    ax.axhline(-65, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.5)

    # Plot cloud of spikes
    trace_color = color if color else COLORS["cloud"]
    for wf in waveforms:
        ax.plot(t_ms, wf, color=trace_color, alpha=0.08, linewidth=0.8)

    ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_components(
    components: np.ndarray,
    output_path: Path,
):
    """Asset 2: Principal components."""
    print(f"  Creating components asset...")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    _setup_schematic_axes(ax)

    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS
    scale = 20.0

    # Reference line at zero
    ax.axhline(0, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.5)

    # Plot PCs with vertical offsets for clarity
    offsets = [0, 0, 0]  # No offset - they're distinct enough
    pc_colors = [COLORS["pc1"], COLORS["pc2"], COLORS["pc3"]]

    for i in range(min(3, len(components))):
        ax.plot(
            t_ms,
            components[i] * scale + offsets[i],
            color=pc_colors[i],
            linewidth=3.0,
            alpha=0.9,
        )

    ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_reconstruction(
    actual_waveform: np.ndarray,
    mean_waveform: np.ndarray,
    components: np.ndarray,
    output_path: Path,
):
    """Asset 3: Actual vs reconstructed spike."""
    print(f"  Creating reconstruction asset...")

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    _setup_schematic_axes(ax)

    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    # Reference line at rest
    ax.axhline(-65, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.5)

    # Compute PCA weights for this spike
    centered = actual_waveform - mean_waveform
    weights = components @ centered  # Project onto PCs

    # Reconstruct with 2 PCs
    reconstructed = (
        mean_waveform + weights[0] * components[0] + weights[1] * components[1]
    )

    # Plot actual (black solid)
    ax.plot(t_ms, actual_waveform, color=COLORS["mean"], linewidth=3.0, alpha=1.0)

    # Plot reconstruction (red dashed)
    ax.plot(
        t_ms,
        reconstructed,
        color=COLORS["recon"],
        linewidth=2.5,
        linestyle="--",
        alpha=0.9,
    )

    ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def main():
    print("=" * 60)
    print("DIAGRAM ASSET GENERATOR")
    print("=" * 60)

    # Load basis data
    basis_path = _OUTPUT_DIR / "basis_data.npz"
    if not basis_path.exists():
        print(f"ERROR: {basis_path} not found. Run ssds_model.py first.")
        raise SystemExit(1)

    print(f"\nLoading basis data from: {basis_path}")
    data = np.load(basis_path)
    mean_waveform = data["mean_waveform"]
    components = data["components"]
    print(f"  Mean shape: {mean_waveform.shape}")
    print(f"  Components shape: {components.shape}")

    # Generate spike data
    print("\nGenerating spike data...")
    cloud_waveforms = generate_spike_cloud(n_samples=1000)
    fatigued_waveform, _ = generate_fatigued_spike()

    # Generate assets
    print("\nGenerating diagram assets...")

    # Gray version
    plot_spike_cloud(
        cloud_waveforms,
        _OUTPUT_DIR / "icon_spike_cloud.png",
    )

    # Alternate color version (deep blue)
    plot_spike_cloud(
        cloud_waveforms,
        _OUTPUT_DIR / "icon_spike_cloud_blue.png",
        color=COLORS["pc2"],
    )

    plot_components(
        components,
        _OUTPUT_DIR / "icon_components.png",
    )

    if fatigued_waveform is not None:
        plot_reconstruction(
            fatigued_waveform,
            mean_waveform,
            components,
            _OUTPUT_DIR / "icon_reconstruction.png",
        )
    else:
        print("  WARNING: Could not generate fatigued spike for reconstruction asset")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Assets saved to: {_OUTPUT_DIR}")
    print("  - icon_spike_cloud.png (gray)")
    print("  - icon_spike_cloud_blue.png (blue)")
    print("  - icon_components.png")
    print("  - icon_reconstruction.png")


if __name__ == "__main__":
    main()
