"""
Fatigue Progression Diagram

Generates a clean two-panel figure showing:
- Top: Input stimulus pulses
- Bottom: Neural response (AIS voltage trace)

No axis labels or ticks for use as diagram assets.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np

from src.simulation import create_pulse_train, DT_MS
from src.ais_simulation import run_2comp_simulation


def generate_fatigue_diagram(
    freq_hz: float = 90.0,
    n_pulses: int = 20,
    output_name: str = "fatigue_diagram.png",
):
    """
    Generate a two-panel fatigue progression diagram.

    Parameters
    ----------
    freq_hz : float
        Stimulation frequency in Hz.
    n_pulses : int
        Number of stimulus pulses.
    output_name : str
        Output filename.
    """
    import matplotlib.pyplot as plt

    # Calculate timing
    isi_ms = 1000.0 / freq_hz
    start_ms = 10.0  # Initial quiet period
    pulse_times = [start_ms + i * isi_ms for i in range(n_pulses)]
    t_end_ms = pulse_times[-1] + 20.0  # Tail after last pulse

    # Create stimulus
    i_stim = create_pulse_train(t_end_ms, pulse_times, dt_ms=DT_MS)
    n_steps = len(i_stim)
    t_ms = np.arange(n_steps) * DT_MS

    # Run 2-compartment simulation
    t_sim, Vs, Va, _, _, _, _ = run_2comp_simulation(
        t_end_ms,
        i_stim,
        dt_ms=DT_MS,
    )

    # Create figure
    fig, (ax_stim, ax_resp) = plt.subplots(
        2,
        1,
        figsize=(10, 4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
    )

    # --- Top panel: Stimulus ---
    ax_stim.fill_between(
        t_ms,
        0,
        i_stim,
        color="#2c3e50",
        alpha=0.8,
        linewidth=0,
    )
    ax_stim.set_ylim(-5, 40)
    ax_stim.set_xlim(0, t_end_ms)

    # Remove all axis elements
    ax_stim.set_xticks([])
    ax_stim.set_yticks([])
    ax_stim.spines["top"].set_visible(False)
    ax_stim.spines["right"].set_visible(False)
    ax_stim.spines["bottom"].set_visible(False)
    ax_stim.spines["left"].set_visible(False)

    # --- Bottom panel: Neural response ---
    ax_resp.plot(
        t_sim,
        Va,
        color="#c0392b",
        linewidth=1.0,
    )
    ax_resp.set_ylim(-80, 50)
    ax_resp.set_xlim(0, t_end_ms)

    # Remove all axis elements
    ax_resp.set_xticks([])
    ax_resp.set_yticks([])
    ax_resp.spines["top"].set_visible(False)
    ax_resp.spines["right"].set_visible(False)
    ax_resp.spines["bottom"].set_visible(False)
    ax_resp.spines["left"].set_visible(False)

    # Save
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    output_path = _OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    print(f"Saved: {output_path}")
    plt.close(fig)

    return output_path


def main():
    print("=" * 60)
    print("FATIGUE PROGRESSION DIAGRAM GENERATOR")
    print("=" * 60)

    # Generate diagrams at different frequencies
    configs = [
        (50, "fatigue_50hz.png"),
        (90, "fatigue_90hz.png"),
        (100, "fatigue_100hz.png"),
    ]

    for freq_hz, filename in configs:
        print(f"\nGenerating {freq_hz} Hz diagram...")
        generate_fatigue_diagram(
            freq_hz=freq_hz,
            n_pulses=20,
            output_name=filename,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
