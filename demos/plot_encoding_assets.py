"""
Generate visual assets for Encoding/Decoding "Exploded View" diagram.

Shows how a spike is decomposed into weighted basis functions:
    Input ≈ Mean + (w1·PC1) + (w2·PC2)

Assets:
1. asset_input_spike.png - The raw fatigued spike (black)
2. asset_mean.png - The mean waveform (gray dashed)
3. asset_pc1_weighted.png - w1 * PC1 (red, may be inverted)
4. asset_pc2_weighted.png - w2 * PC2 (blue)
5. asset_summation.png - Reconstructed spike (black dashed)
6. asset_spike_library.png - Full spike library (multi-color cloud)
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
    generate_protocol_a,
    generate_protocol_c,
)

# Seaborn deep palette
COLORS = {
    "input": "#1a1a1a",  # near black
    "mean": "#8c8c8c",  # neutral gray
    "pc1": "#c44e52",  # deep red
    "pc2": "#4c72b0",  # deep blue
    "pc3": "#55a868",  # deep green
    "recon": "#1a1a1a",  # near black for reconstruction
    "A_fatigue": "#c44e52",  # deep red
    "C_population": "#8c8c8c",  # neutral gray
}


def _setup_schematic_axes(ax):
    """Configure axes for clean schematic look."""
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_fatigued_spike_train() -> list[dict]:
    """Generate Protocol A (100Hz train) and return all spikes."""
    print("  Generating 100Hz fatigue train (20 pulses)...")
    freq_hz = 100
    isi_ms = 1000.0 / freq_hz
    pulse_times = [5.0 + i * isi_ms for i in range(20)]

    spikes = _simulate_and_harvest(
        pulse_times,
        metadata={"protocol": "A_fatigue", "freq_hz": freq_hz},
    )
    return spikes


def plot_sticker(
    t_ms: np.ndarray,
    y_data: np.ndarray,
    output_path: Path,
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 3.0,
    y_range: tuple[float, float] | None = None,
):
    """Generic sticker plot with consistent styling."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    _setup_schematic_axes(ax)

    # Reference line at zero (for weighted components) or rest (for voltages)
    ref_y = 0 if y_range and y_range[0] < -10 else -65
    ax.axhline(ref_y, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.4)

    ax.plot(
        t_ms, y_data, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95
    )

    ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)
    if y_range:
        ax.set_ylim(y_range)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_library_sticker(
    t_ms: np.ndarray,
    spikes: list[dict],
    output_path: Path,
    *,
    y_range: tuple[float, float] | None = None,
):
    """Sticker plot showing the entire spike library overlaid."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    _setup_schematic_axes(ax)

    # Reference line at rest
    # ax.axhline(-65, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.4) 

    # Plot all spikes with protocol colors
    for s in spikes:
        color = COLORS.get(s["protocol"], "#8c8c8c")
        alpha = 0.2 if s["protocol"] == "C_population" else 0.4
        ax.plot(t_ms, s["waveform"], color=color, alpha=alpha, linewidth=0.5)

    ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)
    if y_range:
        ax.set_ylim(y_range)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved library asset: {output_path}")


def main():
    print("=" * 60)
    print("ENCODING ASSETS GENERATOR")
    print("Exploded View: Input ≈ Mean + (w1·PC1) + (w2·PC2)")
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

    # Generate fatigued spike
    print("\nGenerating spike data...")
    spikes = generate_fatigued_spike_train()

    if len(spikes) < 10:
        print(f"ERROR: Only {len(spikes)} spikes generated, need at least 10")
        raise SystemExit(1)

    # Extract the 10th spike (index 9) - the fatigued one
    target_spike = spikes[9]
    input_waveform = target_spike["waveform"]
    print(f"  Using spike #{target_spike['spike_num']} (10th in train)")

    # Calculate PCA weights
    print("\nCalculating PCA weights...")
    centered = input_waveform - mean_waveform
    w1 = np.dot(components[0], centered)
    w2 = np.dot(components[1], centered)
    w3 = np.dot(components[2], centered)
    print(f"  w1 = {w1:.2f}")
    print(f"  w2 = {w2:.2f}")
    print(f"  w3 = {w3:.2f}")

    # Compute weighted components
    pc1_weighted = w1 * components[0]
    pc2_weighted = w2 * components[1]
    pc3_weighted = w3 * components[2]

    # Reconstruction
    reconstructed = mean_waveform + pc1_weighted + pc2_weighted

    # Reconstruction error
    rmse = np.sqrt(np.mean((input_waveform - reconstructed) ** 2))
    print(f"  Reconstruction RMSE (2 PCs): {rmse:.3f} mV")

    # Time axis
    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    # Determine y-ranges for consistency
    voltage_min = min(input_waveform.min(), mean_waveform.min(), reconstructed.min())
    voltage_max = max(input_waveform.max(), mean_waveform.max(), reconstructed.max())
    voltage_range = (voltage_min - 5, voltage_max + 5)

    component_min = min(pc1_weighted.min(), pc2_weighted.min())
    component_max = max(pc1_weighted.max(), pc2_weighted.max())
    component_range = (component_min - 2, component_max + 2)

    # Generate assets
    print("\nGenerating sticker assets...")

    # Asset 1: Input spike (black solid)
    print("  Creating input spike asset...")
    plot_sticker(
        t_ms,
        input_waveform,
        _OUTPUT_DIR / "asset_input_spike.png",
        color=COLORS["input"],
        linestyle="-",
        y_range=voltage_range,
    )

    # Asset 2: Mean waveform (gray dashed)
    print("  Creating mean waveform asset...")
    plot_sticker(
        t_ms,
        mean_waveform,
        _OUTPUT_DIR / "asset_mean.png",
        color=COLORS["mean"],
        linestyle="--",
        y_range=voltage_range,
    )

    # Asset 3: w1 * PC1 (red)
    print("  Creating weighted PC1 asset...")
    plot_sticker(
        t_ms,
        pc1_weighted,
        _OUTPUT_DIR / "asset_pc1_weighted.png",
        color=COLORS["pc1"],
        linestyle="-",
        y_range=component_range,
    )

    # Asset 4: w2 * PC2 (blue)
    print("  Creating weighted PC2 asset...")
    plot_sticker(
        t_ms,
        pc2_weighted,
        _OUTPUT_DIR / "asset_pc2_weighted.png",
        color=COLORS["pc2"],
        linestyle="-",
        y_range=component_range,
    )

    # Asset 5: Reconstruction (black dashed)
    print("  Creating reconstruction asset...")
    plot_sticker(
        t_ms,
        reconstructed,
        _OUTPUT_DIR / "asset_summation.png",
        color=COLORS["recon"],
        linestyle="--",
        y_range=voltage_range,
    )

    # Asset 6: Full Library (multi-color)
    print("  Creating full library asset...")
    # Generate full library for the sticker
    # Note: Using smaller n_samples for C to keep asset generation fast if needed, 
    # but the request asks for "all the spikes that we have generated".
    full_library = (
        generate_protocol_a() + 
        generate_protocol_c(n_samples=200) # 200 is plenty for the visual "cloud"
    )
    plot_library_sticker(
        t_ms,
        full_library,
        _OUTPUT_DIR / "asset_spike_library.png",
        y_range=voltage_range,
    )

    # Print weight info for diagram annotation
    print("\n" + "-" * 60)
    print("WEIGHT VALUES FOR DIAGRAM LABELS")
    print("-" * 60)
    print(f"  Spike time: t = {target_spike['peak_time_ms']:.1f} ms")
    print(f"  Weights: w = [{w1:.1f}, {w2:.1f}, {w3:.1f}]")
    print(f"  (or rounded: w = [{int(round(w1))}, {int(round(w2))}, {int(round(w3))}])")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Assets saved to: {_OUTPUT_DIR}")
    print("  - asset_input_spike.png    (Input spike - black solid)")
    print("  - asset_mean.png           (Mean waveform - gray dashed)")
    print("  - asset_pc1_weighted.png   (w1·PC1 - red)")
    print("  - asset_pc2_weighted.png   (w2·PC2 - blue)")
    print("  - asset_summation.png      (Reconstruction - black dashed)")
    print("  - asset_spike_library.png  (Full library - multi-color)")
    print()
    print("Equation: Input ≈ Mean + (w1·PC1) + (w2·PC2)")
    print(f"          where w1={w1:.1f}, w2={w2:.1f}")


if __name__ == "__main__":
    main()
