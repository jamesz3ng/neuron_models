"""
Refractory Period Stress Test for Event Model

Tests how the EventPropagator handles closely spaced spikes.
1. Absolute Refractory: Does it block inputs within the lockout window (5ms)?
2. Relative Refractory: Does it accurately reconstruct the "stunted" shape
   of the second spike once it is allowed through?
"""

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent / 'src'))

import matplotlib.pyplot as plt
import numpy as np

from event_model import EventPropagator, _generate_hh_signal

# =============================================================================
# Configuration
# =============================================================================

# Test a range of intervals to find the "Cutoff" point
# We expect the model (lockout=5ms) to block spikes with ISI < 5ms
TEST_ISIS = [1, 2, 2.5, 3.0, 4.0, 5.5, 7.0, 10.0, 15.0]

DELAY_MS = 5.0
LOCKOUT_MS = 5.0  # Must match the internal setting in EventPropagator (default 5.0)


def count_spikes(v_trace, threshold=-20.0):
    """Simple threshold crossing counter for validation."""
    crossings = np.diff((v_trace > threshold).astype(int))
    return np.sum(crossings > 0)


def main():
    print("=" * 70)
    print("EVENT MODEL REFRACTORY TEST")
    print(f"Model Lockout: {LOCKOUT_MS} ms")
    print("=" * 70)

    # Initialize model
    prop = EventPropagator(delay_ms=DELAY_MS)

    # Setup plot
    n_rows = (len(TEST_ISIS) + 1) // 2
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(12, 3 * n_rows), sharex=True, sharey=True
    )
    axes = axes.flatten()

    results = []

    for i, isi in enumerate(TEST_ISIS):
        ax = axes[i]

        # 1. Generate Input (Paired Pulse)
        # We assume 2 pulses. Freq = 1000/ISI.
        freq = 1000.0 / isi
        t_ms, v_in = _generate_hh_signal(
            freq_hz=freq, n_pulses=2, pre_ms=5.0, post_ms=15.0
        )

        # Count input spikes (Did the physics even allow a 2nd spike?)
        n_in = count_spikes(v_in)

        # 2. Simulate Propagation
        sim_res = prop.simulate(v_in, t_ms)
        v_out = sim_res["v_out"]
        n_out = sim_res["n_spikes"]

        # 3. Analyze
        # Shift input to align with output for visual comparison
        t_shifted = t_ms + DELAY_MS

        # Determine Status
        status = "UNKNOWN"
        if n_in == 1:
            status = "PHYSICS BLOCK"  # The HH model itself couldn't fire fast enough
            color_st = "gray"
        elif n_in == 2 and n_out == 1:
            status = "MODEL BLOCK"  # Event model enforced lockout
            color_st = "red"
        elif n_in == 2 and n_out == 2:
            status = "PASSED"  # Successful transmission
            color_st = "green"

        results.append((isi, n_in, n_out, status))

        # 4. Plotting
        ax.plot(t_shifted, v_in, "k-", alpha=0.3, label="Input (Delayed)")
        ax.plot(t_ms, v_out, "r--", label="Reconstruction")

        # Add labels
        ax.set_title(f"ISI = {isi} ms | In:{n_in} -> Out:{n_out}", fontweight="bold")
        ax.text(
            0.95,
            0.95,
            status,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=color_st,
            fontweight="bold",
        )

        # Mark the lockout window from the first spike
        if len(sim_res["events"]) > 0:
            t_first = sim_res["events"][0][0] * prop.dt_ms  # First spike arrival time
            ax.axvspan(
                t_first, t_first + LOCKOUT_MS, color="red", alpha=0.1, label="Lockout"
            )

        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", fontsize="small")

    # Formatting
    for ax in axes:
        ax.set_ylim(-80, 50)

    fig.text(0.5, 0.02, "Time (ms)", ha="center")
    fig.text(0.02, 0.5, "Voltage (mV)", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

    output_file = "refractory_test.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved plot to: {output_file}")

    # Print Summary Table
    print("\n" + "-" * 60)
    print(
        f"{'ISI (ms)':<10} | {'Input Spikes':<12} | {'Output Spikes':<13} | {'Result'}"
    )
    print("-" * 60)
    for res in results:
        print(f"{res[0]:<10.1f} | {res[1]:<12} | {res[2]:<13} | {res[3]}")
    print("-" * 60)


if __name__ == "__main__":
    main()
