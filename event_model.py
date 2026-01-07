"""
Event-based Action Potential Propagator

Simulates axonal propagation by:
1. Detecting spikes in source voltage trace
2. Encoding each spike as sparse event (arrival_time + 3 PCA weights)
3. Delaying events by axonal conduction time
4. Reconstructing waveforms at destination

This achieves massive compression while preserving spike shape fidelity.
"""

import numpy as np


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
        """
        Initialize the EventPropagator.
        
        Parameters
        ----------
        basis_file : str
            Path to the NPZ file containing PCA basis data.
        delay_ms : float
            Axonal conduction delay in milliseconds.
        v_rest : float
            Resting membrane potential (mV). Must match training data.
        """
        # Load basis data
        data = np.load(basis_file)
        self.mean_waveform = data["mean_waveform"]  # (window_samples,)
        self.components = data["components"]  # (n_components, window_samples)
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
        
        # Pre-compute for fast encoding (components already in correct shape)
        self.basis_matrix = self.components  # (n_components, window_samples)
        self.n_components = self.components.shape[0]
        
        # Detection parameters
        self.spike_threshold_mv = -20.0
        self.peak_search_ms = 2.0
        self.lockout_ms = 5.0
        self.peak_search_samples = int(self.peak_search_ms / self.dt_ms)
        self.lockout_samples = int(self.lockout_ms / self.dt_ms)
    
    def encode(self, v_window: np.ndarray) -> np.ndarray:
        """
        Encode a spike window into PCA weights.
        
        Parameters
        ----------
        v_window : np.ndarray
            Voltage array of size window_samples, centered on spike peak.
        
        Returns
        -------
        np.ndarray
            Array of n_components weights (typically 3 floats).
        """
        # Subtract mean, project onto components
        centered = v_window - self.mean_waveform
        weights = np.dot(self.basis_matrix, centered)
        return weights
    
    def decode(self, weights: np.ndarray) -> np.ndarray:
        """
        Decode PCA weights back into a spike waveform.
        
        Parameters
        ----------
        weights : np.ndarray
            Array of n_components weights.
        
        Returns
        -------
        np.ndarray
            Reconstructed voltage array of size window_samples.
        """
        reconstruction = self.mean_waveform + np.dot(weights, self.basis_matrix)
        return reconstruction
    
    def simulate(
        self,
        v_source: np.ndarray,
        t_ms: np.ndarray | None = None,
        debug: bool = False,
    ) -> dict:
        """
        Simulate axonal propagation via event encoding/decoding.
        
        Parameters
        ----------
        v_source : np.ndarray
            Source voltage trace (mV).
        t_ms : np.ndarray | None
            Time array (ms). If None, generated from dt_ms.
        debug : bool
            If True, store debug info for first spike.
        
        Returns
        -------
        dict
            Contains:
            - v_out: Reconstructed voltage at destination
            - events: List of (arrival_index, weights) tuples
            - n_spikes: Number of detected spikes
            - compression_ratio: Raw size / Event size
            - debug_info: (if debug=True) dict with first spike alignment data
        """
        n_samples = len(v_source)
        
        if t_ms is None:
            t_ms = np.arange(n_samples) * self.dt_ms
        
        # Step 1: Scan & Encode
        events = []
        debug_info = None
        i = 0
        
        while i < n_samples - 1:
            # Check for upward threshold crossing
            if v_source[i] < self.spike_threshold_mv <= v_source[i + 1]:
                # Find peak within search window
                search_end = min(i + self.peak_search_samples, n_samples)
                peak_idx = i + np.argmax(v_source[i:search_end])
                
                # Extract window centered on peak (with v_rest padding if needed)
                window_start = peak_idx - self.pre_samples
                window_end = peak_idx + self.post_samples
                
                v_window = np.full(self.window_samples, self.v_rest)
                
                # Calculate valid source indices and window indices
                src_start = max(0, window_start)
                src_end = min(n_samples, window_end)
                win_start = max(0, -window_start)
                win_end = win_start + (src_end - src_start)
                
                v_window[win_start:win_end] = v_source[src_start:src_end]
                
                # Encode to weights
                weights = self.encode(v_window)
                
                # Debug: log first spike info
                if debug and len(events) == 0:
                    print(f"\n[DEBUG] First spike:")
                    print(f"  peak_idx: {peak_idx}")
                    print(f"  window_start: {window_start}, window_end: {window_end}")
                    print(f"  src_start: {src_start}, src_end: {src_end}")
                    print(f"  win_start: {win_start}, win_end: {win_end}")
                    print(f"  v_window length: {len(v_window)}")
                    print(f"  v_window peak index: {np.argmax(v_window)}")
                    print(f"  v_window peak value: {np.max(v_window):.2f} mV")
                    print(f"  Expected peak index: {self.pre_samples}")
                    print(f"  Weights: {weights}")
                    print(f"  Weight magnitudes: {np.abs(weights)}")
                    debug_info = {
                        "v_window": v_window.copy(),
                        "peak_idx": peak_idx,
                        "weights": weights.copy(),
                    }
                
                # Store event with arrival time (peak + delay)
                arrival_idx = peak_idx + self.delay_samples
                events.append((arrival_idx, weights))
                
                # Apply lockout
                i = peak_idx + self.lockout_samples
            else:
                i += 1
        
        # Step 2: Reconstruct (with overlap handling)
        v_out = np.full(n_samples, self.v_rest)
        
        for event_idx, (arrival_idx, weights) in enumerate(events):
            # Decode waveform
            v_recon = self.decode(weights)
            
            # Calculate perturbation from rest
            v_delta = v_recon - self.v_rest
            
            # Determine output indices (centered on arrival)
            out_start = arrival_idx - self.pre_samples
            out_end = arrival_idx + self.post_samples
            
            # Truncate at next spike to avoid double-counting overlap
            if event_idx + 1 < len(events):
                next_arrival = events[event_idx + 1][0]
                # Truncate before next spike's pre-window starts
                next_start = next_arrival - self.pre_samples
                if out_end > next_start:
                    out_end = next_start
            
            # Handle boundary clipping
            win_start = max(0, -out_start)
            win_end = win_start + (min(n_samples, out_end) - max(0, out_start))
            out_start = max(0, out_start)
            out_end = min(n_samples, out_end)
            
            # Add perturbation (allows overlapping spikes to sum)
            if out_end > out_start:
                v_out[out_start:out_end] += v_delta[win_start:win_end]
        
        # Calculate compression ratio
        raw_size = n_samples * 8  # float64
        event_size = len(events) * 4 * 8  # 4 float64s per event (1 time + 3 weights)
        compression_ratio = raw_size / event_size if event_size > 0 else float("inf")
        
        result = {
            "v_out": v_out,
            "t_ms": t_ms,
            "events": events,
            "n_spikes": len(events),
            "compression_ratio": compression_ratio,
        }
        
        if debug and debug_info is not None:
            result["debug_info"] = debug_info
        
        return result


# =============================================================================
# Helper: Generate 85Hz fatigue train (minimal version from ssds_model.py)
# =============================================================================

def _generate_fatigue_train(
    freq_hz: float = 85.0,
    n_pulses: int = 10,
    pre_ms: float = 10.0,
    post_ms: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a pulse train and run through HH model.
    
    Returns (t_ms, V) for the fatigue protocol.
    """
    from math import exp as math_exp
    
    # HH parameters
    C_m = 1.0
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    v_rest = -65.0
    
    dt_ms = 0.01
    stim_amplitude = 30.0
    stim_duration_ms = 1.0
    
    # Calculate timing
    isi_ms = 1000.0 / freq_hz
    pulse_times = [pre_ms + i * isi_ms for i in range(n_pulses)]
    t_end_ms = pulse_times[-1] + post_ms
    
    n_time = int(t_end_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms
    
    # Create stimulus array
    i_stim = np.zeros(n_time)
    for t_pulse in pulse_times:
        start_idx = int(t_pulse / dt_ms)
        end_idx = int((t_pulse + stim_duration_ms) / dt_ms)
        end_idx = min(end_idx, n_time)
        if start_idx < n_time:
            i_stim[start_idx:end_idx] = stim_amplitude
    
    # Initialize state
    V_val = v_rest
    m_val = 0.0529
    h_val = 0.5961
    n_val = 0.3177
    
    inv_C_m = 1.0 / C_m
    V_hist = np.zeros(n_time)
    V_hist[0] = V_val
    
    # Integration loop
    for i in range(1, n_time):
        I_Na = g_Na * (m_val**3) * h_val * (V_val - E_Na)
        I_K = g_K * (n_val**4) * (V_val - E_K)
        I_L = g_L * (V_val - E_L)
        
        V_p55 = V_val + 55.0
        V_p40 = V_val + 40.0
        V_p65 = V_val + 65.0
        V_p35 = V_val + 35.0
        
        a_n = -0.01 * V_p55 / (math_exp(V_p55 / -10.0) - 1.0)
        b_n = 0.125 * math_exp(V_p65 / -80.0)
        a_m = -0.1 * V_p40 / (math_exp(V_p40 / -10.0) - 1.0)
        b_m = 4.0 * math_exp(V_p65 / -18.0)
        a_h = 0.07 * math_exp(V_p65 / -20.0)
        b_h = 1.0 / (1.0 + math_exp(V_p35 / -10.0))
        
        dm = a_m * (1.0 - m_val) - b_m * m_val
        dh = a_h * (1.0 - h_val) - b_h * h_val
        dn = a_n * (1.0 - n_val) - b_n * n_val
        
        m_val += dm * dt_ms
        h_val += dh * dt_ms
        n_val += dn * dt_ms
        
        I_stim_val = i_stim[i - 1]
        dV = (I_stim_val - I_Na - I_K - I_L) * inv_C_m
        V_val += dV * dt_ms
        
        V_hist[i] = V_val
    
    return t_ms, V_hist


# =============================================================================
# Signal Generators for Testing
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
    Generate HH signal with configurable parameters.
    
    Returns (t_ms, V).
    """
    from math import exp as math_exp
    
    # HH parameters
    C_m = 1.0
    g_Na = 120.0 * g_na_scale
    g_K = 36.0 * g_k_scale
    g_L = 0.3
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    v_rest = -65.0
    
    dt_ms = 0.01
    stim_amplitude = 30.0
    stim_duration_ms = 1.0
    
    # Calculate timing
    isi_ms = 1000.0 / freq_hz
    pulse_times = [pre_ms + i * isi_ms for i in range(n_pulses)]
    t_end_ms = pulse_times[-1] + post_ms
    
    n_time = int(t_end_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms
    
    # Create stimulus array
    i_stim = np.zeros(n_time)
    for t_pulse in pulse_times:
        start_idx = int(t_pulse / dt_ms)
        end_idx = int((t_pulse + stim_duration_ms) / dt_ms)
        end_idx = min(end_idx, n_time)
        if start_idx < n_time:
            i_stim[start_idx:end_idx] = stim_amplitude
    
    # Compute steady-state gate values for v_init
    def alpha_m(V): return -0.1 * (V + 40) / (math_exp((V + 40) / -10) - 1)
    def beta_m(V): return 4.0 * math_exp((V + 65) / -18)
    def alpha_h(V): return 0.07 * math_exp((V + 65) / -20)
    def beta_h(V): return 1.0 / (1.0 + math_exp((V + 35) / -10))
    def alpha_n(V): return -0.01 * (V + 55) / (math_exp((V + 55) / -10) - 1)
    def beta_n(V): return 0.125 * math_exp((V + 65) / -80)
    
    m_inf = alpha_m(v_init) / (alpha_m(v_init) + beta_m(v_init))
    h_inf = alpha_h(v_init) / (alpha_h(v_init) + beta_h(v_init))
    n_inf = alpha_n(v_init) / (alpha_n(v_init) + beta_n(v_init))
    
    # Initialize state
    V_val = v_init
    m_val = m_inf
    h_val = h_inf
    n_val = n_inf
    
    inv_C_m = 1.0 / C_m
    V_hist = np.zeros(n_time)
    V_hist[0] = V_val
    
    # Integration loop
    for i in range(1, n_time):
        I_Na = g_Na * (m_val**3) * h_val * (V_val - E_Na)
        I_K = g_K * (n_val**4) * (V_val - E_K)
        I_L = g_L * (V_val - E_L)
        
        V_p55 = V_val + 55.0
        V_p40 = V_val + 40.0
        V_p65 = V_val + 65.0
        V_p35 = V_val + 35.0
        
        a_n = -0.01 * V_p55 / (math_exp(V_p55 / -10.0) - 1.0)
        b_n = 0.125 * math_exp(V_p65 / -80.0)
        a_m = -0.1 * V_p40 / (math_exp(V_p40 / -10.0) - 1.0)
        b_m = 4.0 * math_exp(V_p65 / -18.0)
        a_h = 0.07 * math_exp(V_p65 / -20.0)
        b_h = 1.0 / (1.0 + math_exp(V_p35 / -10.0))
        
        dm = a_m * (1.0 - m_val) - b_m * m_val
        dh = a_h * (1.0 - h_val) - b_h * h_val
        dn = a_n * (1.0 - n_val) - b_n * n_val
        
        m_val += dm * dt_ms
        h_val += dh * dt_ms
        n_val += dn * dt_ms
        
        I_stim_val = i_stim[i - 1]
        dV = (I_stim_val - I_Na - I_K - I_L) * inv_C_m
        V_val += dV * dt_ms
        
        V_hist[i] = V_val
    
    return t_ms, V_hist


def _compute_rmse(
    prop: EventPropagator,
    t_ms: np.ndarray,
    v_source: np.ndarray,
    v_out: np.ndarray,
) -> tuple[float, float]:
    """Compute RMSE and max error for spike regions."""
    from scipy.interpolate import interp1d
    
    delay_ms = prop.delay_ms
    t_shifted = t_ms + delay_ms
    
    interp_source = interp1d(t_shifted, v_source, bounds_error=False, fill_value=prop.v_rest)
    v_source_aligned = interp_source(t_ms)
    
    residual = v_out - v_source_aligned
    
    # Only measure in spike regions
    spike_mask = (v_source_aligned > -60) | (v_out > -60)
    if np.any(spike_mask):
        rmse = np.sqrt(np.mean(residual[spike_mask] ** 2))
        max_error = np.max(np.abs(residual[spike_mask]))
    else:
        rmse = 0.0
        max_error = 0.0
    
    return rmse, max_error


# =============================================================================
# Main: Comprehensive Multi-Signal Test
# =============================================================================

def main():
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("EventPropagator - Comprehensive Signal Reconstruction Test")
    print("=" * 70)
    
    # Initialize propagator
    delay_ms = 5.0
    prop = EventPropagator(delay_ms=delay_ms)
    
    print(f"\nPropagator config:")
    print(f"  Window: {prop.window_pre_ms}ms pre + {prop.window_post_ms}ms post = {prop.window_pre_ms + prop.window_post_ms}ms")
    print(f"  Components: {prop.n_components}")
    print(f"  Delay: {delay_ms}ms")
    
    # Define test cases
    test_cases = [
        # (name, generator_kwargs, description)
        ("Low freq (20Hz)", dict(freq_hz=20.0, n_pulses=5), "Isolated spikes, no overlap"),
        ("Medium freq (50Hz)", dict(freq_hz=50.0, n_pulses=10), "Moderate firing rate"),
        ("High freq (75Hz)", dict(freq_hz=75.0, n_pulses=10), "Near refractory limit"),
        ("Extreme freq (85Hz)", dict(freq_hz=85.0, n_pulses=10), "Overlapping windows"),
        ("Single spike", dict(freq_hz=10.0, n_pulses=1), "Minimal case"),
        ("Long burst (50Hz)", dict(freq_hz=50.0, n_pulses=20), "Extended train"),
        ("High gNa (+15%)", dict(freq_hz=50.0, n_pulses=5, g_na_scale=1.15), "Taller spikes"),
        ("Low gNa (-15%)", dict(freq_hz=50.0, n_pulses=5, g_na_scale=0.85), "Shorter spikes"),
        ("High gK (+15%)", dict(freq_hz=50.0, n_pulses=5, g_k_scale=1.15), "Faster repolarization"),
        ("Hyperpolarized (-80mV)", dict(freq_hz=50.0, n_pulses=5, v_init=-80.0), "Super-charged first spike"),
        ("Depolarized (-60mV)", dict(freq_hz=50.0, n_pulses=5, v_init=-60.0), "Reduced Na availability"),
    ]
    
    # Run tests
    results = []
    print("\n" + "-" * 70)
    print(f"{'Test Case':<30} {'Spikes':>7} {'RMSE':>10} {'Max Err':>10} {'Compress':>10}")
    print("-" * 70)
    
    for name, kwargs, description in test_cases:
        # Generate signal
        t_ms, v_source = _generate_hh_signal(**kwargs)
        
        # Simulate propagation
        result = prop.simulate(v_source, t_ms)
        v_out = result["v_out"]
        n_spikes = result["n_spikes"]
        compression = result["compression_ratio"]
        
        # Compute errors
        rmse, max_err = _compute_rmse(prop, t_ms, v_source, v_out)
        
        # Store results
        results.append({
            "name": name,
            "description": description,
            "kwargs": kwargs,
            "n_spikes": n_spikes,
            "rmse": rmse,
            "max_err": max_err,
            "compression": compression,
            "t_ms": t_ms,
            "v_source": v_source,
            "v_out": v_out,
        })
        
        # Print row
        status = "✓" if rmse < 2.0 else "⚠" if rmse < 5.0 else "✗"
        print(f"{status} {name:<28} {n_spikes:>7} {rmse:>9.3f}mV {max_err:>9.3f}mV {compression:>9.1f}x")
    
    print("-" * 70)
    
    # Summary statistics
    rmses = [r["rmse"] for r in results if r["n_spikes"] > 0]
    max_errs = [r["max_err"] for r in results if r["n_spikes"] > 0]
    compressions = [r["compression"] for r in results if r["n_spikes"] > 0]
    
    print(f"\nSummary ({len(rmses)} tests with spikes):")
    print(f"  RMSE:        mean={np.mean(rmses):.3f}mV, max={np.max(rmses):.3f}mV")
    print(f"  Max Error:   mean={np.mean(max_errs):.3f}mV, max={np.max(max_errs):.3f}mV")
    print(f"  Compression: mean={np.mean(compressions):.1f}x, range=[{np.min(compressions):.1f}x, {np.max(compressions):.1f}x]")
    
    # Create multi-panel plot
    n_cases = len(results)
    n_cols = 3
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        t_ms = r["t_ms"]
        v_source = r["v_source"]
        v_out = r["v_out"]
        
        # Shift source for overlay
        t_shifted = t_ms + delay_ms
        
        ax.plot(t_shifted, v_source, "b-", alpha=0.6, linewidth=1, label="Original")
        ax.plot(t_ms, v_out, "r--", alpha=0.8, linewidth=1, label="Reconstructed")
        ax.set_title(f"{r['name']}\nRMSE={r['rmse']:.2f}mV, {r['n_spikes']} spikes", fontsize=9)
        ax.set_ylim(-85, 55)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("V (mV)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")
    
    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("event_propagator_validation.png", dpi=150)
    print(f"\nPlot saved: event_propagator_validation.png")
    plt.close()
    
    # Print pass/fail summary
    passed = sum(1 for r in results if r["rmse"] < 2.0 and r["n_spikes"] > 0)
    warned = sum(1 for r in results if 2.0 <= r["rmse"] < 5.0)
    failed = sum(1 for r in results if r["rmse"] >= 5.0 or r["n_spikes"] == 0)
    
    print(f"\n" + "=" * 70)
    print(f"RESULTS: {passed} passed (✓), {warned} warnings (⚠), {failed} failed (✗)")
    print("=" * 70)


if __name__ == "__main__":
    main()
