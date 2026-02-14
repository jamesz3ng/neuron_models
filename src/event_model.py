"""
Event-based Action Potential Propagator

Simulates axonal propagation by:
1. Detecting spikes in source voltage trace
2. Encoding each spike as sparse event (arrival_time + 3 PCA weights)
3. Delaying events by axonal conduction time
4. Reconstructing waveforms at destination
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# Default basis file path relative to this module
_DEFAULT_BASIS_FILE = Path(__file__).parent.parent / "output" / "basis_data.npz"

# =============================================================================
# Core Model Class (The Novel Part)
# =============================================================================


class EventPropagator:
    """
    Encodes spikes as sparse events, delays them, and reconstructs at destination.
    Uses PCA basis from basis_data.npz for encoding/decoding.
    """

    def __init__(
        self,
        basis_file: str | Path | None = None,
        delay_ms: float = 5.0,
        v_rest: float = -65.0,
    ):
        if basis_file is None:
            basis_file = _DEFAULT_BASIS_FILE
        # Load basis data
        data = np.load(basis_file)
        self.mean_waveform = data["mean_waveform"]
        self.components = data["components"]
        self.dt_ms = float(data["dt_ms"])
        self.window_pre_ms = float(data["window_pre_ms"])
        self.window_post_ms = float(data["window_post_ms"])
        self.window_samples = int(data["window_samples"])

        # Derived parameters
        self.pre_samples = int(self.window_pre_ms / self.dt_ms)
        self.post_samples = int(self.window_post_ms / self.dt_ms)
        self.delay_ms = delay_ms
        self.delay_samples = int(delay_ms / self.dt_ms)
        self.v_rest = v_rest

        # --- Reconstruction taper ---
        # The HH afterhyperpolarization (AHP) doesn't recover to v_rest within
        # the 8ms post-peak window: it's still ~7mV below rest at window end.
        # We build a taper applied at reconstruction time that blends the decoded
        # waveform into v_rest at the window boundaries.
        
        fade_in_samples = int(3.0 / self.dt_ms)  # 3ms fade-in
        fade_out_samples = int(5.0 / self.dt_ms)  # 5ms fade-out
        n_points = self.window_samples

        self._recon_taper = np.ones(n_points)
        if fade_in_samples > 0 and n_points > fade_in_samples:
            self._recon_taper[:fade_in_samples] = np.linspace(0.0, 1.0, fade_in_samples)
        if fade_out_samples > 0 and n_points > fade_out_samples:
            self._recon_taper[-fade_out_samples:] = np.linspace(
                1.0, 0.0, fade_out_samples
            )

        # Pre-compute for fast encoding
        self.basis_matrix = self.components
        self.n_components = self.components.shape[0]

        # Detection parameters
        self.spike_threshold_mv = -20.0
        self.peak_search_samples = int(2.0 / self.dt_ms)  # 2ms window
        self.lockout_samples = int(5.0 / self.dt_ms)  # 5ms lockout

    def encode(self, v_window: np.ndarray) -> np.ndarray:
        return np.dot(self.basis_matrix, v_window - self.mean_waveform)

    def decode(self, weights: np.ndarray) -> np.ndarray:
        return self.mean_waveform + np.dot(weights, self.basis_matrix)

    def _extract_events(
        self, v_source: np.ndarray, *, delay_samples: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Detect threshold crossings and encode spikes as delayed events."""
        n_samples = len(v_source)
        applied_delay = self.delay_samples if delay_samples is None else delay_samples

        events = []
        i = 0
        while i < n_samples - 1:
            if v_source[i] < self.spike_threshold_mv <= v_source[i + 1]:
                search_end = min(i + self.peak_search_samples, n_samples)
                peak_idx = i + np.argmax(v_source[i:search_end])

                win_start = peak_idx - self.pre_samples
                win_end = peak_idx + self.post_samples
                v_window = np.full(self.window_samples, self.v_rest)

                src_start = max(0, win_start)
                src_end = min(n_samples, win_end)
                dest_start = max(0, -win_start)
                dest_end = dest_start + (src_end - src_start)
                v_window[dest_start:dest_end] = v_source[src_start:src_end]

                weights = self.encode(v_window)
                events.append((peak_idx + applied_delay, weights))
                i = peak_idx + self.lockout_samples
            else:
                i += 1

        return events

    def _reconstruct_events(
        self,
        events: list[tuple[int, np.ndarray]],
        n_samples: int,
        *,
        mode: str = "overwrite",
    ) -> np.ndarray:
        """
        Reconstruct waveform from events.

        Parameters
        ----------
        events : list[tuple[int, np.ndarray]]
            List of (arrival_idx, PCA weights).
        n_samples : int
            Output waveform length.
        mode : str
            "overwrite" for last-event dominance, "sum" for additive perturbations.
        """
        if mode not in {"overwrite", "sum"}:
            raise ValueError("mode must be 'overwrite' or 'sum'")

        v_out = np.full(n_samples, self.v_rest)
        for arrival_idx, weights in events:
            v_recon = self.decode(weights)

            # Apply reconstruction taper: blend decoded waveform into v_rest
            # at window boundaries to eliminate step artifacts from the AHP tail
            perturbation = v_recon - self.v_rest
            v_tapered = self.v_rest + perturbation * self._recon_taper

            start = arrival_idx - self.pre_samples
            end = arrival_idx + self.post_samples
            src_start = max(0, -start)
            src_end = self.window_samples - max(0, end - n_samples)
            dest_start = max(0, start)
            dest_end = min(n_samples, end)

            if dest_end <= dest_start:
                continue

            if mode == "overwrite":
                v_out[dest_start:dest_end] = v_tapered[src_start:src_end]
            else:
                v_out[dest_start:dest_end] += v_tapered[src_start:src_end] - self.v_rest

        return v_out

    def _coerce_sources(self, v_sources: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """Normalize converging source traces to shape (n_sources, n_samples)."""
        if isinstance(v_sources, list):
            if len(v_sources) == 0:
                raise ValueError("v_sources must contain at least one source trace")
            sources = np.vstack([np.asarray(v, dtype=float) for v in v_sources])
        else:
            sources = np.asarray(v_sources, dtype=float)
            if sources.ndim == 1:
                sources = sources[np.newaxis, :]
            elif sources.ndim != 2:
                raise ValueError("v_sources must be 1D, 2D, or list of 1D arrays")
        return sources

    def simulate(self, v_source: np.ndarray, t_ms: np.ndarray | None = None) -> dict:
        n_samples = len(v_source)
        if t_ms is None:
            t_ms = np.arange(n_samples) * self.dt_ms

        events = self._extract_events(v_source)
        v_out = self._reconstruct_events(events, n_samples, mode="overwrite")

        # Stats
        raw_size = n_samples * 8
        event_size = len(events) * (1 + self.n_components) * 8
        ratio = raw_size / event_size if event_size > 0 else float("inf")

        return {
            "v_out": v_out,
            "t_ms": t_ms,
            "n_spikes": len(events),
            "compression_ratio": ratio,
            "events": events,
        }

    def simulate_converging(
        self,
        v_sources: np.ndarray | list[np.ndarray],
        t_ms: np.ndarray | None = None,
        *,
        tau_ms: float = 0.01,
        input_gain: float = 1.0,
        delays_ms: float | list[float] | np.ndarray | None = None,
        fusion_mode: str = "lpf",
        baseline_center: bool = True,
    ) -> dict:
        """
        Simulate convergence of multiple upstream sources at one downstream node.

        Parameters
        ----------
        v_sources : np.ndarray | list[np.ndarray]
            Source traces with shape (n_sources, n_samples), (n_samples,), or list.
        t_ms : np.ndarray | None
            Time vector in ms. If None, inferred from basis dt.
        tau_ms : float
            Membrane time constant for the leaky integrator (ms).
            Controls how quickly the downstream voltage tracks upstream
            input. Small tau = fast tracking, large tau = heavy smoothing.
        input_gain : float
            Scalar gain applied to upstream perturbations before integration.
        delays_ms : float | list[float] | np.ndarray | None
            Optional per-source delays. If scalar, same delay for all sources.
            If None, uses self.delay_ms for every source.
        fusion_mode : str
            'lpf' for leaky-integrator fusion of all arrivals (recommended),
            'last_event' for legacy overwrite dominance.
        baseline_center : bool
            Deprecated. LPF mode now always operates in perturbation-from-rest
            space. Kept for backward compatibility.

        Returns
        -------
        dict
            Includes fused output, per-source arrivals, event metadata, and stats.
            'i_in' contains the perturbation from v_rest (diagnostic).
        """
        if fusion_mode not in {"lpf", "last_event"}:
            raise ValueError("fusion_mode must be 'lpf' or 'last_event'")
        if tau_ms <= 0:
            raise ValueError("tau_ms must be > 0")

        sources = self._coerce_sources(v_sources)
        n_sources, n_samples = sources.shape

        if t_ms is None:
            t_ms = np.arange(n_samples) * self.dt_ms
        elif len(t_ms) != n_samples:
            raise ValueError("t_ms length must match source trace length")

        # Resolve per-source delays
        if delays_ms is None:
            delay_samples = np.full(n_sources, self.delay_samples, dtype=int)
        elif np.isscalar(delays_ms):
            delay_samples = np.full(
                n_sources, int(round(float(delays_ms) / self.dt_ms)), dtype=int
            )
        else:
            delays_arr = np.asarray(delays_ms, dtype=float)
            if len(delays_arr) != n_sources:
                raise ValueError("delays_ms length must match number of sources")
            delay_samples = np.round(delays_arr / self.dt_ms).astype(int)

        events_by_source: list[list[tuple[int, np.ndarray]]] = []
        arrivals_by_source = np.full((n_sources, n_samples), self.v_rest, dtype=float)
        merged_events: list[tuple[int, np.ndarray]] = []

        for src_idx in range(n_sources):
            events = self._extract_events(
                sources[src_idx], delay_samples=int(delay_samples[src_idx])
            )
            events_by_source.append(events)
            merged_events.extend(events)
            arrivals_by_source[src_idx] = self._reconstruct_events(
                events, n_samples, mode="overwrite"
            )

        if fusion_mode == "last_event":
            merged_events.sort(key=lambda e: e[0])
            v_out = self._reconstruct_events(merged_events, n_samples, mode="overwrite")
            i_in = np.zeros(n_samples)
        else:
            # Leaky integrator: tau * dV/dt = -(V - V_rest) + gain * I_upstream
            # Discrete form: V[i] = V[i-1] + (dt/tau)*(-( V[i-1] - V_rest) + drive)
            v_out = np.full(n_samples, self.v_rest, dtype=float)
            i_in = np.zeros(n_samples, dtype=float)
            alpha = self.dt_ms / tau_ms  # dimensionless decay factor

            for i in range(1, n_samples):
                # Sum upstream perturbations from rest, rectified: only
                # depolarising (positive) perturbations contribute.  The
                # afterhyperpolarisation is intrinsic to the presynaptic
                # neuron and should not propagate as negative drive.
                per_source = arrivals_by_source[:, i - 1] - self.v_rest
                upstream = np.sum(np.maximum(per_source, 0.0))

                # Scaled upstream drive
                drive = input_gain * upstream

                # Leaky integrator update
                v_prev = v_out[i - 1] - self.v_rest
                v_out[i] = self.v_rest + v_prev + alpha * (-v_prev + drive)

                # Track perturbation for diagnostics
                i_in[i] = v_out[i] - self.v_rest

        n_spikes_by_source = [len(events) for events in events_by_source]
        n_spikes_total = int(sum(n_spikes_by_source))
        raw_size = n_sources * n_samples * 8
        event_size = n_spikes_total * (1 + self.n_components) * 8
        ratio = raw_size / event_size if event_size > 0 else float("inf")

        return {
            "v_out": v_out,
            "i_in": i_in,
            "t_ms": t_ms,
            "n_sources": n_sources,
            "n_spikes": n_spikes_total,
            "n_spikes_by_source": n_spikes_by_source,
            "compression_ratio": ratio,
            "events_by_source": events_by_source,
            "arrivals_by_source": arrivals_by_source,
            "tau_ms": float(tau_ms),
            "input_gain": float(input_gain),
            "fusion_mode": fusion_mode,
            "baseline_center": bool(baseline_center),
        }

    def print_memory_stats(self, spikes_per_100ms: int = 5) -> None:
        """Print memory comparison: DDE delay-line vs Event-Based model."""
        # DDE stores full waveform buffer (same window as event model)
        dde_bytes = self.window_samples * 8  # float64

        # Event-based stores per-spike: arrival_time + n_components weights
        event_bytes_per_spike = (1 + self.n_components) * 8  # float64
        event_bytes = spikes_per_100ms * event_bytes_per_spike

        # Shared overhead (amortized across all axons)
        shared_mean = self.mean_waveform.nbytes
        shared_components = self.components.nbytes
        shared_total = shared_mean + shared_components

        # Break-even: how many axons before shared overhead is amortized
        per_axon_savings = dde_bytes - event_bytes
        breakeven_axons = (
            int(shared_total / per_axon_savings) if per_axon_savings > 0 else 0
        )

        # Memory reduction ratio
        reduction = dde_bytes / event_bytes if event_bytes > 0 else float("inf")

        # Print table
        print("=" * 80)
        print("MEMORY COMPARISON: Per-Axon AP Storage")
        print("=" * 80)
        print()
        print("DDE Delay Line (stores full AP waveform buffer):")
        print(
            f"  - Window duration: {self.window_pre_ms + self.window_post_ms:.1f} ms "
            f"(pre: {self.window_pre_ms:.1f} ms, post: {self.window_post_ms:.1f} ms)"
        )
        print(f"  - Samples: {self.window_samples:,} @ {self.dt_ms} ms dt")
        print(f"  - Memory per axon: {dde_bytes:,} bytes ({dde_bytes / 1024:.1f} KB)")
        print()
        print("Event-Based Model (sparse spike encoding):")
        print(
            f"  - Per spike: {event_bytes_per_spike} bytes "
            f"(1 arrival_time + {self.n_components} PCA weights)"
        )
        print(f"  - Assumed spikes per 100ms: {spikes_per_100ms}")
        print(f"  - Memory per axon: {event_bytes:,} bytes")
        print()
        print("Shared Overhead (amortized across all axons):")
        print(f"  - Mean waveform: {shared_mean:,} bytes")
        print(f"  - PCA components ({self.n_components}): {shared_components:,} bytes")
        print(
            f"  - Total shared: {shared_total:,} bytes ({shared_total / 1024:.1f} KB)"
        )
        print()
        print(f"Break-even point: {breakeven_axons:,} axons")
        print(f"Memory Reduction per Axon: {reduction:.0f}x")
        print("=" * 80)


# =============================================================================
# Helper: AIS Signal Generator
# =============================================================================


def _generate_ais_signal(
    freq_hz: float = 50.0,
    n_pulses: int = 10,
    g_na_s_scale: float = 1.0,
    g_k_s_scale: float = 1.0,
    pre_ms: float = 10.0,
    post_ms: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unified generator for all test protocols.

    Uses the 2-compartment (Soma + AIS) model so that the spike shape
    matches the PCA basis that EventPropagator was trained on.
    Returns the AIS voltage (Va) as the axonal output signal.
    """
    from .ais_simulation import run_2comp_simulation
    from .simulation import DT_MS, create_pulse_train

    # Stimulus setup
    isi_ms = 1000.0 / freq_hz
    t_end_ms = pre_ms + (n_pulses * isi_ms) + post_ms
    pulse_times = [pre_ms + k * isi_ms for k in range(n_pulses)]
    i_stim = create_pulse_train(t_end_ms, pulse_times, dt_ms=DT_MS)

    t_ms, _Vs, Va, *_ = run_2comp_simulation(
        t_end_ms,
        i_stim,
        dt_ms=DT_MS,
        g_na_s_scale=g_na_s_scale,
        g_k_s_scale=g_k_s_scale,
    )

    return t_ms, Va


# =============================================================================
# Main
# =============================================================================


def main():
    import matplotlib.pyplot as plt

    print("=" * 60 + "\nEventPropagator Verification\n" + "=" * 60)

    prop = EventPropagator(delay_ms=5.0)

    # Test Cases — all use 2-comp AIS source matching PCA basis
    cases = [
        ("Fatigue Train (85Hz)", dict(freq_hz=85.0, n_pulses=10)),
        ("Low Freq (20Hz)", dict(freq_hz=20.0, n_pulses=5)),
        ("Scaled gNa (0.9x)", dict(freq_hz=50.0, n_pulses=5, g_na_s_scale=0.9)),
    ]

    for name, params in cases:
        print(f"\nTesting: {name}")
        t_ms, v_src = _generate_ais_signal(**params)
        res = prop.simulate(v_src, t_ms)

        # Error Calc (Aligned)
        t_shift = t_ms + prop.delay_ms
        v_aligned = interp1d(
            t_shift, v_src, bounds_error=False, fill_value=prop.v_rest
        )(t_ms)
        err = v_aligned - res["v_out"]
        mask = (v_aligned > -60) | (res["v_out"] > -60)  # Only check spikes
        rmse = np.sqrt(np.mean(err[mask] ** 2)) if np.any(mask) else 0.0

        print(f"  RMSE: {rmse:.3f} mV | Compression: {res['compression_ratio']:.1f}x")

    # Print memory comparison
    print()
    prop.print_memory_stats()

    print("\nVerification Complete.")


if __name__ == "__main__":
    main()
