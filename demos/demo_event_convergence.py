"""
Event-Based Convergence Demo

Compares legacy last-event dominance vs LPF fusion for multiple upstream sources
using the EventPropagator basis-function pipeline.

Also includes a single-source equivalence test showing that with one upstream
neuron, the LPF output converges to the last_event reconstruction as tau -> 0.
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

from src.event_model import EventPropagator
from src.simulation import DT_MS, create_pulse_train
from src.ais_simulation import run_2comp_simulation


def _count_spikes(v_trace: np.ndarray, threshold_mv: float = -20.0) -> int:
    crossings = np.diff((v_trace >= threshold_mv).astype(int))
    return int(np.sum(crossings > 0))


def _make_sources(t_end_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate overlapping upstream AIS traces for convergence testing.

    Uses the 2-compartment (Soma + AIS) model so that the spike shape
    matches the PCA basis that EventPropagator was trained on.
    The AIS voltage (Va) is used as the axonal output signal.
    """
    pulse_sets = [
        [6.0, 20.0, 34.0],
        [7.2, 21.2, 35.2],
        [8.4, 22.4, 36.4],
    ]

    sources = []
    t_ms = None
    for pulses in pulse_sets:
        i_stim = create_pulse_train(t_end_ms, pulses, dt_ms=DT_MS)
        t_i, _Vs, Va, *_ = run_2comp_simulation(t_end_ms, i_stim, dt_ms=DT_MS)
        if t_ms is None:
            t_ms = t_i
        sources.append(Va)

    return t_ms, np.vstack(sources)


def _plot_multi_source(
    t_ms: np.ndarray,
    res_last: dict,
    lpf_results: dict[float, dict],
    tau_values: list[float],
    input_gain: float,
):
    """Plot the original 3-source convergence comparison (Figure 1)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1 = axes[0]
    arrivals = lpf_results[tau_values[0]]["arrivals_by_source"]
    for i in range(arrivals.shape[0]):
        ax1.plot(t_ms, arrivals[i], linewidth=1.2, label=f"Arrival source {i + 1}")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Per-Source Event-Based Arrivals at Downstream Node")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2 = axes[1]
    ax2.plot(
        t_ms,
        res_last["v_out"],
        color="black",
        linewidth=2.2,
        linestyle="--",
        label="Legacy last_event",
    )
    for tau_ms in tau_values:
        ax2.plot(
            t_ms,
            lpf_results[tau_ms]["v_out"],
            linewidth=2,
            label=f"LPF tau={tau_ms:.1f} ms",
        )
    ax2.axhline(-20.0, color="gray", linestyle=":", linewidth=1.0)
    ax2.set_ylabel("Fused V_out (mV)")
    ax2.set_title("Convergence Fusion: Last-Event vs LPF")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    ax3 = axes[2]
    for tau_ms in tau_values:
        ax3.plot(
            t_ms,
            lpf_results[tau_ms]["i_in"],
            linewidth=2,
            label=f"LPF perturbation tau={tau_ms:.1f} ms",
        )
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("V - V_rest (mV)")
    ax3.set_title(f"LPF Perturbation from Rest (input_gain={input_gain:.2f})")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    plt.tight_layout()
    output_file = _OUTPUT_DIR / "event_convergence_lpf_vs_last_event.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_file}")


def _plot_single_source(t_ms: np.ndarray, v_single: np.ndarray, prop: EventPropagator):
    """Single-source equivalence test (Figure 2).

    With only one upstream neuron, the LPF output should converge to the
    last_event reconstruction as tau -> 0, since there is no competing
    input to integrate.  Any residual difference comes from the LPF's
    half-wave rectification (AHP clipped to zero) and temporal smoothing.
    """
    import matplotlib.pyplot as plt

    # Wrap as (1, N) for simulate_converging
    v_1src = v_single[np.newaxis, :]

    # Reference: last_event (pure event reconstruction, no filtering)
    res_last = prop.simulate_converging(
        v_1src,
        t_ms=t_ms,
        fusion_mode="last_event",
    )

    # LPF at several tau values — input_gain=1.0 (no scaling)
    tau_sweep = [0.01, 0.1]
    lpf_results: dict[float, dict] = {}
    for tau in tau_sweep:
        lpf_results[tau] = prop.simulate_converging(
            v_1src,
            t_ms=t_ms,
            fusion_mode="lpf",
            tau_ms=tau,
            input_gain=1.0,
        )

    # --- Figure ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Panel 1: waveform overlay
    ax1 = axes[0]
    ax1.plot(
        t_ms,
        res_last["v_out"],
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="last_event (reference)",
    )
    cmap = plt.cm.viridis
    for idx, tau in enumerate(tau_sweep):
        color = cmap(idx / max(len(tau_sweep) - 1, 1))
        ax1.plot(
            t_ms,
            lpf_results[tau]["v_out"],
            linewidth=1.8,
            color=color,
            label=f"LPF tau={tau} ms",
        )
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title("Single-Source Equivalence: last_event vs LPF at Various tau")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize="small")

    # Panel 2: difference from last_event
    ax2 = axes[1]
    for idx, tau in enumerate(tau_sweep):
        color = cmap(idx / max(len(tau_sweep) - 1, 1))
        diff = lpf_results[tau]["v_out"] - res_last["v_out"]
        rmse = float(np.sqrt(np.mean(diff**2)))
        peak_err = float(np.max(np.abs(diff)))
        ax2.plot(
            t_ms,
            diff,
            linewidth=1.5,
            color=color,
            label=f"tau={tau} ms  RMSE={rmse:.2f} mV  peak={peak_err:.1f} mV",
        )
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_ylabel("V_lpf - V_last_event (mV)")
    ax2.set_title("Difference: LPF minus last_event")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize="small")

    # Panel 3: perturbation from rest (LPF diagnostic)
    ax3 = axes[2]
    for idx, tau in enumerate(tau_sweep):
        color = cmap(idx / max(len(tau_sweep) - 1, 1))
        ax3.plot(
            t_ms,
            lpf_results[tau]["i_in"],
            linewidth=1.5,
            color=color,
            label=f"tau={tau} ms",
        )
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("V - V_rest (mV)")
    ax3.set_title("LPF Perturbation from Rest (single source, gain=1.0)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize="small")

    plt.tight_layout()
    output_file = _OUTPUT_DIR / "event_convergence_single_source.png"
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_file}")

    # Print diagnostics
    print("\nSingle-source equivalence test:")
    print(f"  last_event spikes: {_count_spikes(res_last['v_out'])}")
    for tau in tau_sweep:
        diff = lpf_results[tau]["v_out"] - res_last["v_out"]
        rmse = float(np.sqrt(np.mean(diff**2)))
        peak_err = float(np.max(np.abs(diff)))
        print(
            f"  LPF tau={tau:>5.2f} ms:  RMSE={rmse:.3f} mV,  "
            f"peak_error={peak_err:.2f} mV,  "
            f"spikes={_count_spikes(lpf_results[tau]['v_out'])}"
        )


def main():
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("EVENT MODEL CONVERGENCE DEMO")
    print("=" * 70)

    t_end_ms = 50.0
    tau_values = [1.0, 2.0, 5.0]
    input_gain = 0.6

    t_ms, v_sources = _make_sources(t_end_ms)
    prop = EventPropagator(delay_ms=5.0)

    # --- Multi-source convergence (original test) ---
    res_last = prop.simulate_converging(
        v_sources,
        t_ms=t_ms,
        fusion_mode="last_event",
        tau_ms=2.0,
    )

    lpf_results: dict[float, dict] = {}
    for tau_ms in tau_values:
        lpf_results[tau_ms] = prop.simulate_converging(
            v_sources,
            t_ms=t_ms,
            tau_ms=tau_ms,
            input_gain=input_gain,
            fusion_mode="lpf",
        )

    _plot_multi_source(t_ms, res_last, lpf_results, tau_values, input_gain)

    print(f"n_sources: {res_last['n_sources']}")
    print(f"n_spikes_by_source: {res_last['n_spikes_by_source']}")
    print()
    print("Output spike counts:")
    print(f"  last_event: {_count_spikes(res_last['v_out'])}")
    for tau_ms in tau_values:
        n_sp = _count_spikes(lpf_results[tau_ms]["v_out"])
        peak_v = float(np.max(lpf_results[tau_ms]["v_out"]))
        peak_i = float(np.max(np.abs(lpf_results[tau_ms]["i_in"])))
        print(
            f"  lpf tau={tau_ms:>4.1f} ms: n_spikes={n_sp}, "
            f"V_peak={peak_v:.3f} mV, perturbation_peak={peak_i:.3f} mV"
        )

    # --- Single-source equivalence test ---
    print()
    print("-" * 70)
    print("SINGLE-SOURCE EQUIVALENCE TEST")
    print("-" * 70)
    _plot_single_source(t_ms, v_sources[0], prop)

    print("=" * 70)


if __name__ == "__main__":
    main()
