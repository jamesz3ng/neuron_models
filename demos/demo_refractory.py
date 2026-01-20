import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np

from src.fast_model import simulate_fast_model


def _generate_spike_train(
    t_ms: np.ndarray,
    spike_times_ms: list[float],
    spike_width_ms: float,
    peak_mv: float,
    v_rest: float,
) -> np.ndarray:
    """
    Generate a synthetic spike train with AP-like waveforms.

    Each spike is modeled as a Gaussian pulse for simplicity.
    """
    v = np.full_like(t_ms, v_rest, dtype=float)

    for t_spike in spike_times_ms:
        # Gaussian spike shape
        spike = (peak_mv - v_rest) * np.exp(
            -((t_ms - t_spike) ** 2) / (2 * spike_width_ms**2)
        )
        v = np.maximum(v, v_rest + spike)

    return v


def main():
    import matplotlib.pyplot as plt

    # Configuration
    T_ms = 50.0
    dt_ms = 0.01  # Fine resolution to capture spike shapes
    t_ms = np.arange(0, T_ms, dt_ms)

    v_rest = -65.0
    peak_mv = 30.0
    spike_width_ms = 0.5  # Narrow spikes (~1ms FWHM)

    # Generate 5 spikes at 2ms intervals (500 Hz - physiologically impossible)
    spike_times = [10.0, 12.0, 14.0, 16.0, 18.0]

    v_input = _generate_spike_train(t_ms, spike_times, spike_width_ms, peak_mv, v_rest)

    # Run fast_model with refractory filtering
    refractory_ms = 5.0
    delay_ms = 2.0  # Small delay for visualization

    result = simulate_fast_model(
        v_input=v_input,
        t_ms_input=t_ms,
        delay_ms=delay_ms,
        v_rest=v_rest,
        refractory_period_ms=refractory_ms,
        spike_threshold_mv=-20.0,
    )

    v_output = result["V"]
    blocked = result["blocked_count"]

    # Print summary
    print("=" * 60)
    print("REFRACTORY PERIOD DEMONSTRATION")
    print("=" * 60)
    print(f"Input: {len(spike_times)} spikes at {spike_times} ms")
    print(f"Spike separation: 2 ms (500 Hz)")
    print(f"Refractory period: {refractory_ms} ms")
    print(f"Delay: {delay_ms} ms")
    print("-" * 60)
    print(f"Spikes blocked: {blocked}")
    print(f"Spikes passed: {len(spike_times) - blocked}")
    print("=" * 60)
    print("\nExpected: Spikes 1 & 4 pass; Spikes 2, 3, 5 blocked")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: Input
    ax1.plot(t_ms, v_input, "b-", linewidth=1.5, label="Input (500 Hz train)")
    ax1.axhline(
        -20, color="gray", linestyle="--", alpha=0.5, label="Threshold (-20 mV)"
    )
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title(f"Input: {len(spike_times)} spikes at 2ms intervals (500 Hz)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-80, 50)

    # Mark spike times
    for i, t_spike in enumerate(spike_times):
        ax1.axvline(t_spike, color="blue", alpha=0.3, linestyle=":")
        ax1.text(t_spike, 40, f"S{i + 1}", ha="center", fontsize=9)

    # Bottom: Output
    ax2.plot(
        t_ms,
        v_output,
        "r-",
        linewidth=1.5,
        label=f"Output (after {refractory_ms}ms refractory)",
    )
    ax2.axhline(-20, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Voltage (mV)")
    ax2.set_title(
        f"Output: {len(spike_times) - blocked} spikes passed, {blocked} blocked"
    )
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-80, 50)

    # Mark which spikes passed/blocked
    expected_passed = [0, 3]  # Indices of spikes that should pass
    for i, t_spike in enumerate(spike_times):
        color = "green" if i in expected_passed else "red"
        label = "PASS" if i in expected_passed else "BLOCK"
        ax2.axvline(t_spike + delay_ms, color=color, alpha=0.3, linestyle=":")
        ax2.text(t_spike + delay_ms, 40, label, ha="center", fontsize=8, color=color)

    plt.tight_layout()
    output_file = _OUTPUT_DIR / "refractory_demo.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved plot to: {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    main()
