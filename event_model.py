"""
Event-based Action Potential Propagator

Simulates axonal propagation by:
1. Detecting spikes in source voltage trace
2. Encoding each spike as sparse event (arrival_time + 3 PCA weights)
3. Delaying events by axonal conduction time
4. Reconstructing waveforms at destination
"""

from math import exp as math_exp

import numpy as np
from scipy.interpolate import interp1d

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
        basis_file: str = "basis_data.npz",
        delay_ms: float = 5.0,
        v_rest: float = -65.0,
    ):
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

        # --- TAPERING FIX ---
        # Force endpoints to match v_rest smoothly to prevent "square step" artifacts.
        # This applies a fade-in/fade-out window to the basis functions.
        taper_len = 50  # 0.5 ms taper (assuming 0.01ms dt)
        n_points = self.mean_waveform.shape[0]

        if n_points > 2 * taper_len:
            # Create taper window (0->1 ... 1->0)
            taper = np.ones(n_points)
            taper[:taper_len] = np.linspace(0.0, 1.0, taper_len)
            taper[-taper_len:] = np.linspace(1.0, 0.0, taper_len)

            # 1. Taper Mean Waveform
            # Calculate perturbation from rest, taper it, then restore rest
            perturbation = self.mean_waveform - self.v_rest
            self.mean_waveform = self.v_rest + (perturbation * taper)

            # 2. Taper Components (they represent variance, should decay to 0)
            for i in range(self.components.shape[0]):
                self.components[i] *= taper

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

    def simulate(self, v_source: np.ndarray, t_ms: np.ndarray | None = None) -> dict:
        n_samples = len(v_source)
        if t_ms is None:
            t_ms = np.arange(n_samples) * self.dt_ms

        # Step 1: Scan & Encode
        events = []
        i = 0
        while i < n_samples - 1:
            if v_source[i] < self.spike_threshold_mv <= v_source[i + 1]:
                # Find peak
                search_end = min(i + self.peak_search_samples, n_samples)
                peak_idx = i + np.argmax(v_source[i:search_end])

                # Extract window (padded)
                win_start = peak_idx - self.pre_samples
                win_end = peak_idx + self.post_samples

                v_window = np.full(self.window_samples, self.v_rest)

                # Copy logic handling boundaries
                src_start = max(0, win_start)
                src_end = min(n_samples, win_end)
                dest_start = max(0, -win_start)
                dest_end = dest_start + (src_end - src_start)

                v_window[dest_start:dest_end] = v_source[src_start:src_end]

                # Encode & Store
                weights = self.encode(v_window)
                events.append((peak_idx + self.delay_samples, weights))

                i = peak_idx + self.lockout_samples
            else:
                i += 1

        # Step 2: Reconstruct (direct overwrite - last spike wins)
        v_out = np.full(n_samples, self.v_rest)

        for arrival_idx, weights in events:
            v_recon = self.decode(weights)

            # Calculate destination indices (handling boundaries)
            start = arrival_idx - self.pre_samples
            end = arrival_idx + self.post_samples

            src_start = max(0, -start)
            src_end = self.window_samples - max(0, end - n_samples)
            dest_start = max(0, start)
            dest_end = min(n_samples, end)

            # Direct overwrite: paste absolute voltage (last spike wins)
            if dest_end > dest_start:
                v_out[dest_start:dest_end] = v_recon[src_start:src_end]

        # Stats
        raw_size = n_samples * 8
        event_size = len(events) * 4 * 8
        ratio = raw_size / event_size if event_size > 0 else float("inf")

        return {
            "v_out": v_out,
            "t_ms": t_ms,
            "n_spikes": len(events),
            "compression_ratio": ratio,
            "events": events,
        }


# =============================================================================
# Helper: Universal HH Signal Generator
# =============================================================================


def _generate_hh_signal(
    freq_hz: float = 50.0,
    n_pulses: int = 10,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
    v_init: float = -65.0,
    pre_ms: float = 10.0,
    post_ms: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unified generator for all test protocols.
    Replaces both _generate_fatigue_train and the manual loops.
    """
    # Parameters
    C_m, g_L, E_Na, E_K, E_L = 1.0, 0.3, 50.0, -77.0, -54.387
    g_Na = 120.0 * g_na_scale
    g_K = 36.0 * g_k_scale
    dt_ms = 0.01

    # Stimulus setup
    isi_ms = 1000.0 / freq_hz
    t_end_ms = pre_ms + (n_pulses * isi_ms) + post_ms
    n_time = int(t_end_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms

    i_stim = np.zeros(n_time)
    for k in range(n_pulses):
        t_pulse = pre_ms + k * isi_ms
        start = int(t_pulse / dt_ms)
        end = int((t_pulse + 1.0) / dt_ms)  # 1ms duration
        if start < n_time:
            i_stim[start : min(end, n_time)] = 30.0  # Amplitude

    # Rate functions (Steady state calc)
    def rates(v):
        vp = v + 0.001 if abs(v + 40) < 0.001 or abs(v + 55) < 0.001 else v
        am = -0.1 * (vp + 40) / (math_exp(-(vp + 40) / 10) - 1)
        bm = 4.0 * math_exp(-(vp + 65) / 18)
        ah = 0.07 * math_exp(-(vp + 65) / 20)
        bh = 1.0 / (1.0 + math_exp(-(vp + 35) / 10))
        an = -0.01 * (vp + 55) / (math_exp(-(vp + 55) / 10) - 1)
        bn = 0.125 * math_exp(-(vp + 65) / 80)
        return am, bm, ah, bh, an, bn

    # Initialization
    am, bm, ah, bh, an, bn = rates(v_init)
    m, h, n = am / (am + bm), ah / (ah + bh), an / (an + bn)
    v = v_init

    v_hist = np.zeros(n_time)
    v_hist[0] = v

    # Loop
    for i in range(1, n_time):
        am, bm, ah, bh, an, bn = rates(v)
        m += (am * (1 - m) - bm * m) * dt_ms
        h += (ah * (1 - h) - bh * h) * dt_ms
        n += (an * (1 - n) - bn * n) * dt_ms

        i_ion = (
            (g_Na * m**3 * h * (v - E_Na))
            + (g_K * n**4 * (v - E_K))
            + (g_L * (v - E_L))
        )

        v += ((i_stim[i - 1] - i_ion) / C_m) * dt_ms
        v_hist[i] = v

    return t_ms, v_hist


# =============================================================================
# Main
# =============================================================================


def main():
    import matplotlib.pyplot as plt

    print("=" * 60 + "\nEventPropagator Verification\n" + "=" * 60)

    prop = EventPropagator(delay_ms=5.0)

    # Test Cases
    cases = [
        ("Fatigue Train (85Hz)", dict(freq_hz=85.0, n_pulses=10)),
        ("Low Freq (20Hz)", dict(freq_hz=20.0, n_pulses=5)),
        ("Hyperpolarized", dict(freq_hz=50.0, n_pulses=5, v_init=-80.0)),
    ]

    for name, params in cases:
        print(f"\nTesting: {name}")
        t_ms, v_src = _generate_hh_signal(**params)
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

    print("\nVerification Complete.")


if __name__ == "__main__":
    main()
