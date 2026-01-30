"""
Generate additional visual assets for diagrams.

Assets:
1. asset_square_wave.png - Idealized square pulse stimulus
2. asset_soma_ap.png - Normal soma action potential (HH model)
3. asset_ais_ap.png - Axon Initial Segment action potential (sharper, faster)
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

from src.simulation import create_pulse_train, DT_MS
from src.ais_simulation import run_2comp_simulation

# Seaborn deep palette
COLORS = {
    "stimulus": "#c44e52",  # deep red
    "soma": "#55a868",  # deep green
    "ais": "#4c72b0",  # deep blue
}


def _setup_schematic_axes(ax):
    """Configure axes for clean schematic look."""
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_sticker(
    t_ms: np.ndarray,
    y_data: np.ndarray,
    output_path: Path,
    *,
    color: str,
    linestyle: str = "-",
    linewidth: float = 3.0,
    y_range: tuple[float, float] | None = None,
    ref_line: float | None = None,
):
    """Generic sticker plot with consistent styling."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    _setup_schematic_axes(ax)

    # Reference line
    if ref_line is not None:
        ax.axhline(ref_line, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.4)

    ax.plot(
        t_ms, y_data, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95
    )

    if y_range:
        ax.set_ylim(y_range)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def generate_square_wave(
    t_end_ms: float = 15.0,
    pulse_start_ms: float = 3.0,
    pulse_duration_ms: float = 8.0,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an idealized square wave pulse."""
    n_steps = int(t_end_ms / DT_MS)
    t_ms = np.arange(n_steps) * DT_MS

    wave = np.zeros(n_steps)
    start_idx = int(pulse_start_ms / DT_MS)
    end_idx = int((pulse_start_ms + pulse_duration_ms) / DT_MS)
    wave[start_idx:end_idx] = amplitude

    return t_ms, wave


def generate_soma_and_ais_ap() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Soma and AIS action potentials using 2-compartment model.
    Returns: (t_ms, V_soma, V_ais)
    """
    print("  Running 2-compartment simulation...")

    # Single pulse stimulus
    t_end_ms = 20.0
    pulse_times = [5.0]
    i_stim = create_pulse_train(t_end_ms, pulse_times)

    # Run simulation
    t_ms, Vs, Va, _, _, _, _ = run_2comp_simulation(
        t_end_ms,
        i_stim,
        dt_ms=DT_MS,
    )

    return t_ms, Vs, Va


def extract_spike_window(
    t_ms: np.ndarray,
    V: np.ndarray,
    window_pre_ms: float = 2.0,
    window_post_ms: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a window around the spike peak."""
    # Find peak
    peak_idx = np.argmax(V)

    # Calculate window indices
    pre_points = int(window_pre_ms / DT_MS)
    post_points = int(window_post_ms / DT_MS)

    start_idx = max(0, peak_idx - pre_points)
    end_idx = min(len(V), peak_idx + post_points)

    # Extract and re-center time axis
    t_window = t_ms[start_idx:end_idx] - t_ms[peak_idx]
    V_window = V[start_idx:end_idx]

    return t_window, V_window


def main():
    print("=" * 60)
    print("ADDITIONAL ASSETS GENERATOR")
    print("=" * 60)

    # Asset 1: Square wave
    print("\nGenerating square wave...")
    t_sq, wave = generate_square_wave()
    plot_sticker(
        t_sq,
        wave,
        _OUTPUT_DIR / "asset_square_wave.png",
        color=COLORS["stimulus"],
        linewidth=5.0,
        y_range=(-0.2, 1.3),
    )

    # Generate Soma and AIS action potentials
    print("\nGenerating action potentials...")
    t_full, V_soma, V_ais = generate_soma_and_ais_ap()

    # Extract spike windows
    t_soma, V_soma_window = extract_spike_window(t_full, V_soma)
    t_ais, V_ais_window = extract_spike_window(t_full, V_ais)

    # Determine common y-range for both
    v_min = min(V_soma_window.min(), V_ais_window.min())
    v_max = max(V_soma_window.max(), V_ais_window.max())
    voltage_range = (v_min - 5, v_max + 5)

    print(f"  Soma peak: {V_soma_window.max():.1f} mV")
    print(f"  AIS peak:  {V_ais_window.max():.1f} mV")

    # Asset 2: Soma AP
    print("\nGenerating soma AP asset...")
    plot_sticker(
        t_soma,
        V_soma_window,
        _OUTPUT_DIR / "asset_soma_ap.png",
        color=COLORS["soma"],
        linewidth=4.5,
        y_range=voltage_range,
    )

    # Asset 3: AIS AP
    print("\nGenerating AIS AP asset...")
    plot_sticker(
        t_ais,
        V_ais_window,
        _OUTPUT_DIR / "asset_ais_ap.png",
        color=COLORS["ais"],
        linewidth=4.5,
        y_range=voltage_range,
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Assets saved to: {_OUTPUT_DIR}")
    print("  - asset_square_wave.png   (Idealized square pulse - gold)")
    print("  - asset_soma_ap.png       (Soma action potential - red)")
    print("  - asset_ais_ap.png        (AIS action potential - blue)")


if __name__ == "__main__":
    main()
