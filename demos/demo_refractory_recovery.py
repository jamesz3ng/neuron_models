"""
Refractory Period Recovery Diagram

Generates a multi-panel figure showing paired-pulse recovery at different ISIs.
Each panel shows:
- Top: Input stimulus (two pulses)
- Bottom: Neural response

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


def generate_single_panel(
    ax_stim,
    ax_resp,
    isi_ms: float,
    t_end_ms: float = 50.0,
    first_pulse_ms: float = 10.0,
):
    """
    Generate stimulus and response for a single ISI panel.

    Parameters
    ----------
    ax_stim : matplotlib.axes.Axes
        Axes for stimulus plot.
    ax_resp : matplotlib.axes.Axes
        Axes for response plot.
    isi_ms : float
        Inter-spike interval in ms.
    t_end_ms : float
        Total simulation duration.
    first_pulse_ms : float
        Time of first pulse.
    """
    # Create paired pulse stimulus
    pulse_times = [first_pulse_ms, first_pulse_ms + isi_ms]
    i_stim = create_pulse_train(t_end_ms, pulse_times, dt_ms=DT_MS)
    n_steps = len(i_stim)
    t_ms = np.arange(n_steps) * DT_MS

    # Run simulation
    t_sim, Vs, Va, _, _, _, _ = run_2comp_simulation(
        t_end_ms,
        i_stim,
        dt_ms=DT_MS,
    )

    # --- Stimulus panel ---
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

    # Remove axis elements
    ax_stim.set_xticks([])
    ax_stim.set_yticks([])
    for spine in ax_stim.spines.values():
        spine.set_visible(False)

    # --- Response panel ---
    ax_resp.plot(
        t_sim,
        Va,
        color="#c0392b",
        linewidth=1.0,
    )
    ax_resp.set_ylim(-80, 50)
    ax_resp.set_xlim(0, t_end_ms)

    # Remove axis elements
    ax_resp.set_xticks([])
    ax_resp.set_yticks([])
    for spine in ax_resp.spines.values():
        spine.set_visible(False)


def generate_refractory_diagram(
    isi_values: list[float] | None = None,
    output_name: str = "refractory_recovery.png",
):
    """
    Generate multi-panel refractory recovery diagram.

    Parameters
    ----------
    isi_values : list[float] | None
        List of ISI values to show. Defaults to [3, 5, 10, 20] ms.
    output_name : str
        Output filename.
    """
    import matplotlib.pyplot as plt

    if isi_values is None:
        isi_values = [3.0, 5.0, 7.0, 8.0, 10.0, 20.0]

    n_panels = len(isi_values)
    t_end_ms = 50.0

    # Create figure with 2 rows per panel (stimulus + response)
    fig, axes = plt.subplots(
        2,
        n_panels,
        figsize=(3 * n_panels, 4),
        gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05, "wspace": 0.1},
    )

    # Handle single panel case
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    for i, isi_ms in enumerate(isi_values):
        ax_stim = axes[0, i]
        ax_resp = axes[1, i]

        generate_single_panel(
            ax_stim,
            ax_resp,
            isi_ms=isi_ms,
            t_end_ms=t_end_ms,
        )

    # Save
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    output_path = _OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=200, bbox_inches="tight", transparent=True)
    print(f"Saved: {output_path}")
    plt.close(fig)

    return output_path


def main():
    print("=" * 60)
    print("REFRACTORY RECOVERY DIAGRAM GENERATOR")
    print("=" * 60)

    # Default ISI values showing recovery progression
    print("\nGenerating refractory recovery diagram...")
    print(
        "ISI values: 3ms (blocked), 5ms (attenuated), 10ms (recovering), 20ms (recovered)"
    )

    generate_refractory_diagram(
        isi_values=[3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 20.0],
        output_name="refractory_recovery.png",
    )

    # Also generate individual panels for flexibility
    print("\nGenerating individual ISI panels...")
    for isi in [3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 20.0]:
        generate_refractory_diagram(
            isi_values=[isi],   
            output_name=f"refractory_isi_{int(isi)}ms.png",
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
