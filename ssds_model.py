"""
Spike Shape Decomposition System (SSDS)

Generates a library of diverse Action Potential shapes and performs PCA
to determine if complex, history-dependent spike shapes can be compressed
into a small basis set (Mean + 2-3 principal components).

Protocols:
- A (Fatigue Spectrum): 10-pulse trains at [50, 100, 130] Hz - captures frequency adaptation
- B (Refractory Curve): Paired pulses with ISI 2.0-20.0ms (0.2ms steps) - fine-grained recovery
- C (Population Heterogeneity): 1000 single pulses with g_Na/g_K ±15% variation

Target: ~1,100 spikes to ensure robust PCA capturing full AP shape space.

Goal: Prove that reconstruction error is negligible with just 2-3 components,
enabling compact representation of history-dependent spike shapes.
"""

from math import exp as math_exp

import numpy as np
from sklearn.decomposition import PCA

# =============================================================================
# Configuration
# =============================================================================

# HH Model defaults (from hh_model.py)
DEFAULT_PARAMS = {
    "C_m": 1.0,  # nF/mm^2
    "g_Na": 120.0,  # uS/cm^2
    "g_K": 36.0,
    "g_L": 0.3,
    "E_Na": 50.0,  # mV
    "E_K": -77.0,
    "E_L": -54.387,
    "v_rest": -65.0,
}

# Simulation parameters
DT_MS = 0.01  # Time step (ms) - 0.01ms for 1000 points per 10ms window
STIM_AMPLITUDE = 30.0  # nA/mm^2 (increased from 20.0 to prevent dropouts)
STIM_DURATION_MS = 1.0  # Duration of each stimulus pulse

# Spike extraction parameters
WINDOW_PRE_MS = 2.0  # ms before peak
WINDOW_POST_MS = 12.0  # ms after peak
WINDOW_TOTAL_MS = WINDOW_PRE_MS + WINDOW_POST_MS  # 10ms total
WINDOW_POINTS = int(WINDOW_TOTAL_MS / DT_MS)  # 1000 points
PRE_PEAK_POINTS = int(WINDOW_PRE_MS / DT_MS)  # 200 points
POST_PEAK_POINTS = int(WINDOW_POST_MS / DT_MS)  # 800 points

# Spike detection threshold
SPIKE_THRESHOLD_MV = -20.0


# =============================================================================
# Custom HH Simulator (single compartment with custom I_stim array)
# =============================================================================


def run_custom_hh(
    t_end_ms: float,
    dt_ms: float,
    i_stim_array: np.ndarray,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
    *,
    v_init: float | None = None,
    m_init: float | None = None,
    h_init: float | None = None,
    n_init: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run HH model with custom stimulus current array.

    Copied integration loop from hh_model.py but uses provided i_stim_array
    and scaled conductances. Single compartment (point neuron) only.

    Parameters
    ----------
    t_end_ms : float
        Simulation duration in ms.
    dt_ms : float
        Time step in ms.
    i_stim_array : np.ndarray
        Stimulus current at each time step (same length as n_time).
    g_na_scale : float
        Scale factor for g_Na (1.0 = default 120.0).
    g_k_scale : float
        Scale factor for g_K (1.0 = default 36.0).
    v_init : float | None
        Initial membrane voltage (default: v_rest = -65mV).
    m_init : float | None
        Initial m gate value (default: 0.0529).
    h_init : float | None
        Initial h gate value (default: 0.5961).
    n_init : float | None
        Initial n gate value (default: 0.3177).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (t_ms, V) - time array and voltage array
    """
    # Parameters
    C_m = DEFAULT_PARAMS["C_m"]
    g_Na = DEFAULT_PARAMS["g_Na"] * g_na_scale
    g_K = DEFAULT_PARAMS["g_K"] * g_k_scale
    g_L = DEFAULT_PARAMS["g_L"]
    E_Na = DEFAULT_PARAMS["E_Na"]
    E_K = DEFAULT_PARAMS["E_K"]
    E_L = DEFAULT_PARAMS["E_L"]
    v_rest = DEFAULT_PARAMS["v_rest"]

    inv_C_m = 1.0 / C_m

    n_time = int(t_end_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms

    # Ensure i_stim_array matches n_time
    if len(i_stim_array) != n_time:
        raise ValueError(
            f"i_stim_array length ({len(i_stim_array)}) != n_time ({n_time})"
        )

    # Initialize state variables (use defaults or provided values)
    V_val = v_init if v_init is not None else v_rest
    m_val = m_init if m_init is not None else 0.0529
    h_val = h_init if h_init is not None else 0.5961
    n_val = n_init if n_init is not None else 0.3177

    # History storage
    V_hist = np.zeros(n_time)
    V_hist[0] = V_val

    # Integration loop (scalar path for efficiency)
    for i in range(1, n_time):
        # Ionic currents
        I_Na = g_Na * (m_val**3) * h_val * (V_val - E_Na)
        I_K = g_K * (n_val**4) * (V_val - E_K)
        I_L = g_L * (V_val - E_L)

        # Inlined rate functions using math.exp (scalar)
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

        # Voltage update (no diffusion for single compartment)
        I_stim = i_stim_array[i - 1]
        dV = (I_stim - I_Na - I_K - I_L) * inv_C_m
        V_val += dV * dt_ms

        V_hist[i] = V_val

    return t_ms, V_hist


# =============================================================================
# Stimulus Generation
# =============================================================================


def _create_pulse_train(
    t_end_ms: float,
    dt_ms: float,
    pulse_times_ms: list[float],
    pulse_duration_ms: float,
    amplitude: float,
) -> np.ndarray:
    """Create stimulus array with pulses at specified times."""
    n_time = int(t_end_ms / dt_ms)
    i_stim = np.zeros(n_time)

    for t_pulse in pulse_times_ms:
        start_idx = int(t_pulse / dt_ms)
        end_idx = int((t_pulse + pulse_duration_ms) / dt_ms)
        end_idx = min(end_idx, n_time)
        if start_idx < n_time:
            i_stim[start_idx:end_idx] = amplitude

    return i_stim


# =============================================================================
# Spike Extraction and Alignment
# =============================================================================


def _find_spike_peaks(
    V: np.ndarray, threshold_mv: float = SPIKE_THRESHOLD_MV
) -> list[int]:
    """
    Find indices of spike peaks in voltage trace.

    Returns indices where V crosses threshold upward and reaches local max.
    """
    peaks = []
    in_spike = False
    spike_start = 0
    max_v = -np.inf
    max_idx = 0

    for i in range(len(V)):
        if V[i] >= threshold_mv and not in_spike:
            # Spike onset
            in_spike = True
            spike_start = i
            max_v = V[i]
            max_idx = i
        elif in_spike and V[i] >= threshold_mv:
            # During spike - track maximum
            if V[i] > max_v:
                max_v = V[i]
                max_idx = i
        elif in_spike and V[i] < threshold_mv:
            # Spike ended - record peak
            peaks.append(max_idx)
            in_spike = False
            max_v = -np.inf

    return peaks


def _extract_aligned_spike(
    V: np.ndarray,
    peak_idx: int,
    pre_points: int = PRE_PEAK_POINTS,
    post_points: int = POST_PEAK_POINTS,
) -> np.ndarray | None:
    """
    Extract spike waveform aligned to peak.

    Returns None if window extends beyond array bounds.
    """
    start_idx = peak_idx - pre_points
    end_idx = peak_idx + post_points

    if start_idx < 0 or end_idx > len(V):
        return None

    return V[start_idx:end_idx].copy()


# =============================================================================
# Protocol A: Fatigue Spectrum (10-pulse trains at multiple frequencies)
# =============================================================================


def generate_protocol_a() -> list[dict]:
    """
    Generate fatigued spikes from 10-pulse trains at multiple frequencies.

    Frequencies: [75, 80, 85] Hz to capture fatigue spectrum.
    - 75Hz: Stable baseline with ~4% adaptation, all spikes fire
    - 80Hz: Stronger adaptation ~11%, some spike loss
    - 85Hz: Extreme "stunted" spikes ~11%, near failure threshold

    Each frequency produces 10 pulses showing progressive adaptation.
    """
    frequencies_hz = [75, 80, 85]
    print(f"Protocol A: Fatigue Spectrum (10 pulses at {frequencies_hz} Hz)...")

    all_spikes = []

    for freq_hz in frequencies_hz:
        isi_ms = 1000.0 / freq_hz  # Convert Hz to ISI in ms
        n_pulses = 10

        # 10 pulses starting at 5ms
        pulse_times = [5.0 + i * isi_ms for i in range(n_pulses)]

        # Simulation duration: last pulse + 15ms for final spike
        t_end_ms = pulse_times[-1] + 15.0

        # Create stimulus
        i_stim = _create_pulse_train(
            t_end_ms, DT_MS, pulse_times, STIM_DURATION_MS, STIM_AMPLITUDE
        )

        # Run simulation
        t_ms, V = run_custom_hh(t_end_ms, DT_MS, i_stim)

        # Find and extract spikes
        peaks = _find_spike_peaks(V)

        for i, peak_idx in enumerate(peaks):
            waveform = _extract_aligned_spike(V, peak_idx)
            if waveform is not None:
                all_spikes.append(
                    {
                        "waveform": waveform,
                        "protocol": "A_fatigue",
                        "freq_hz": freq_hz,
                        "spike_num": i + 1,
                        "peak_idx": peak_idx,
                        "peak_time_ms": peak_idx * DT_MS,
                    }
                )

        print(f"  {freq_hz}Hz: Found {len(peaks)} spikes")

    print(f"  Total extracted: {len(all_spikes)} spikes")
    return all_spikes


# =============================================================================
# Protocol B: Refractory Curve (paired pulses, fine-grained ISI)
# =============================================================================


def generate_protocol_b() -> list[dict]:
    """
    Generate refractory recovery spikes from paired-pulse protocol.

    ISI varies from 2.0ms to 20.0ms in 0.2ms steps (91 values).
    Extract the SECOND spike (showing incomplete recovery).
    """
    print("Protocol B: Refractory Curve (paired pulses, ISI 2.0-20.0ms, step 0.2ms)...")

    # ISI from 2.0 to 20.0 in 0.2ms steps
    isi_values = np.arange(2.0, 20.2, 0.2)  # 91 values
    spikes = []

    for isi_ms in isi_values:
        # Two pulses: first at 5ms, second at 5+ISI
        pulse_times = [5.0, 5.0 + isi_ms]
        t_end_ms = pulse_times[-1] + 15.0

        # Create stimulus
        i_stim = _create_pulse_train(
            t_end_ms, DT_MS, pulse_times, STIM_DURATION_MS, STIM_AMPLITUDE
        )

        # Run simulation
        t_ms, V = run_custom_hh(t_end_ms, DT_MS, i_stim)

        # Find peaks
        peaks = _find_spike_peaks(V)

        # Extract the SECOND spike if it exists
        if len(peaks) >= 2:
            waveform = _extract_aligned_spike(V, peaks[1])
            if waveform is not None:
                spikes.append(
                    {
                        "waveform": waveform,
                        "protocol": "B_refractory",
                        "isi_ms": isi_ms,
                        "peak_idx": peaks[1],
                        "peak_time_ms": peaks[1] * DT_MS,
                    }
                )

    print(
        f"  Extracted {len(spikes)} second-spike waveforms (from {len(isi_values)} ISI values)"
    )
    return spikes


# =============================================================================
# Protocol C: Population Heterogeneity (g_Na, g_K ±15%, 1000 samples)
# =============================================================================


def generate_protocol_c(n_samples: int = 2000, seed: int = 42) -> list[dict]:
    """
    Generate population variance spikes with g_Na/g_K variation.

    Each simulation has g_Na and g_K randomly scaled by ±15%.
    Default n_samples=1000 for robust coverage of heterogeneity space.
    """
    print(f"Protocol C: Population Heterogeneity ({n_samples} samples, ±15%)...")

    np.random.seed(seed)
    spikes = []

    for i in range(n_samples):
        # Random scaling: uniform in [0.85, 1.15]
        g_na_scale = np.random.uniform(0.85, 1.15)
        g_k_scale = np.random.uniform(0.85, 1.15)

        # Single pulse at 5ms
        pulse_times = [5.0]
        t_end_ms = 25.0

        # Create stimulus
        i_stim = _create_pulse_train(
            t_end_ms, DT_MS, pulse_times, STIM_DURATION_MS, STIM_AMPLITUDE
        )

        # Run simulation with scaled conductances
        t_ms, V = run_custom_hh(t_end_ms, DT_MS, i_stim, g_na_scale, g_k_scale)

        # Find and extract spike
        peaks = _find_spike_peaks(V)

        if len(peaks) >= 1:
            waveform = _extract_aligned_spike(V, peaks[0])
            if waveform is not None:
                spikes.append(
                    {
                        "waveform": waveform,
                        "protocol": "C_population",
                        "g_na_scale": g_na_scale,
                        "g_k_scale": g_k_scale,
                        "peak_idx": peaks[0],
                        "peak_time_ms": peaks[0] * DT_MS,
                    }
                )

    print(f"  Extracted {len(spikes)} spikes")
    return spikes


# =============================================================================
# Protocol D: Hyperpolarization Rebound ("Super-Charged" Spikes)
# =============================================================================


def generate_protocol_d(n_samples: int = 50) -> list[dict]:
    """
    Generate "super-charged" spikes from hyperpolarized initial states.

    When a neuron is held at hyperpolarized potentials (e.g., -80mV to -90mV),
    the sodium inactivation gate (h) de-inactivates fully (h -> ~1.0).
    This results in spikes that are TALLER and SHARPER than standard spikes.

    This protocol balances the "fatigue" spikes by forcing PCA to learn
    how to GROW a spike, not just shrink it.

    Implementation approaches:
    1. Direct initialization: Set V_init and h_init to hyperpolarized steady-state
    2. Pre-hyperpolarization: Inject negative current for 20ms before the pulse

    We use approach 1 for cleaner control over initial conditions.
    """
    print("Protocol D: Hyperpolarization Rebound (super-charged spikes)...")

    # Hyperpolarization levels to test
    # More negative = more h de-inactivation = taller spike
    v_hyper_levels = [-70, -75, -80, -85, -90]

    # Steady-state gate values at different voltages (from HH equations)
    # h_inf = alpha_h / (alpha_h + beta_h)
    # At V=-65: h_inf ~ 0.596
    # At V=-80: h_inf ~ 0.92
    # At V=-90: h_inf ~ 0.98
    def h_inf(V: float) -> float:
        """Compute steady-state h at voltage V."""
        alpha_h = 0.07 * math_exp((V + 65.0) / -20.0)
        beta_h = 1.0 / (1.0 + math_exp((V + 35.0) / -10.0))
        return alpha_h / (alpha_h + beta_h)

    def n_inf(V: float) -> float:
        """Compute steady-state n at voltage V."""
        alpha_n = -0.01 * (V + 55.0) / (math_exp((V + 55.0) / -10.0) - 1.0)
        beta_n = 0.125 * math_exp((V + 65.0) / -80.0)
        return alpha_n / (alpha_n + beta_n)

    def m_inf(V: float) -> float:
        """Compute steady-state m at voltage V."""
        alpha_m = -0.1 * (V + 40.0) / (math_exp((V + 40.0) / -10.0) - 1.0)
        beta_m = 4.0 * math_exp((V + 65.0) / -18.0)
        return alpha_m / (alpha_m + beta_m)

    spikes = []
    samples_per_level = n_samples // len(v_hyper_levels)

    for v_hyper in v_hyper_levels:
        # Compute steady-state gate values at this hyperpolarized voltage
        h_init = h_inf(v_hyper)
        n_init_val = n_inf(v_hyper)
        m_init = m_inf(v_hyper)

        for _ in range(samples_per_level):
            # Single pulse at 5ms
            pulse_times = [5.0]
            t_end_ms = 25.0

            # Create stimulus
            i_stim = _create_pulse_train(
                t_end_ms, DT_MS, pulse_times, STIM_DURATION_MS, STIM_AMPLITUDE
            )

            # Run simulation with hyperpolarized initial conditions
            t_ms, V = run_custom_hh(
                t_end_ms,
                DT_MS,
                i_stim,
                v_init=v_hyper,
                m_init=m_init,
                h_init=h_init,
                n_init=n_init_val,
            )

            # Find and extract spike
            peaks = _find_spike_peaks(V)

            if len(peaks) >= 1:
                waveform = _extract_aligned_spike(V, peaks[0])
                if waveform is not None:
                    spikes.append(
                        {
                            "waveform": waveform,
                            "protocol": "D_hyperpol",
                            "v_hyper": v_hyper,
                            "h_init": h_init,
                            "peak_idx": peaks[0],
                            "peak_time_ms": peaks[0] * DT_MS,
                            "peak_voltage": V[peaks[0]],
                        }
                    )

        print(
            f"  V_hyper={v_hyper}mV: h_init={h_init:.3f}, extracted {samples_per_level} spikes"
        )

    print(f"  Total extracted: {len(spikes)} super-charged spikes")
    return spikes


# =============================================================================
# PCA Analysis
# =============================================================================


def run_pca_analysis(spikes: list[dict], n_components: int = 3) -> dict:
    """
    Run PCA on spike waveforms.

    Returns dict with:
    - mean: mean waveform
    - components: PC1, PC2, PC3
    - explained_variance_ratio: variance explained by each component
    - pca: fitted PCA object
    - X: spike matrix (n_spikes, n_points)
    - weights: projection weights for each spike
    """
    # Stack waveforms into matrix
    X = np.vstack([s["waveform"] for s in spikes])
    print(f"\nPCA Analysis: {X.shape[0]} spikes, {X.shape[1]} time points")

    # Fit PCA
    pca = PCA(n_components=n_components)
    weights = pca.fit_transform(X)

    # Print explained variance
    print(f"Explained Variance Ratio:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {var * 100:.2f}%")
    print(
        f"  Total (first {n_components}): {sum(pca.explained_variance_ratio_) * 100:.2f}%"
    )

    return {
        "mean": pca.mean_,
        "components": pca.components_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
        "X": X,
        "weights": weights,
    }


def reconstruct_spike(
    pca_result: dict,
    spike_idx: int,
    n_components: int = 2,
) -> np.ndarray:
    """Reconstruct a spike using mean + n_components PCs."""
    mean = pca_result["mean"]
    components = pca_result["components"]
    weights = pca_result["weights"][spike_idx]

    reconstructed = mean.copy()
    for i in range(n_components):
        reconstructed += weights[i] * components[i]

    return reconstructed


# =============================================================================
# Validation: Comprehensive Reconstruction Testing
# =============================================================================


def validate_reconstructions(spikes: list[dict], pca_result: dict) -> dict:
    """
    Comprehensive validation of reconstruction quality across spike types.

    Tests reconstruction on:
    - Protocol A: First spike (fresh), middle spike (adapting), last spike (fatigued)
    - Protocol B: Short ISI (partial recovery), medium ISI, long ISI (full recovery)
    - Protocol C: Extreme g_Na/g_K combinations (corners of parameter space)
    - Worst-case: Spike with highest reconstruction error

    Returns dict with validation results for each test case.
    """
    X = pca_result["X"]
    mean = pca_result["mean"]
    weights = pca_result["weights"]
    components = pca_result["components"]

    # Compute all reconstruction errors
    recon_2pc = mean + weights[:, :2] @ components[:2]
    errors_2pc = np.sqrt(np.mean((X - recon_2pc) ** 2, axis=1))

    recon_3pc = mean + weights[:, :3] @ components[:3]
    errors_3pc = np.sqrt(np.mean((X - recon_3pc) ** 2, axis=1))

    validation = {"test_cases": [], "summary": {}}

    # Helper to add test case
    def add_case(name: str, idx: int, description: str):
        spike = spikes[idx]
        validation["test_cases"].append(
            {
                "name": name,
                "description": description,
                "idx": idx,
                "protocol": spike["protocol"],
                "rmse_2pc": errors_2pc[idx],
                "rmse_3pc": errors_3pc[idx],
                "weights": weights[idx].tolist(),
                "spike_meta": {k: v for k, v in spike.items() if k != "waveform"},
            }
        )

    # --- Protocol A: Fatigue progression ---
    fatigue_indices = [i for i, s in enumerate(spikes) if s["protocol"] == "A_fatigue"]
    if fatigue_indices:
        # Group by frequency
        freq_groups = {}
        for idx in fatigue_indices:
            freq = spikes[idx].get("freq_hz", 0)
            if freq not in freq_groups:
                freq_groups[freq] = []
            freq_groups[freq].append(idx)

        # For each frequency, get first and last spike
        for freq in sorted(freq_groups.keys()):
            indices = freq_groups[freq]
            if len(indices) >= 1:
                add_case(
                    f"A_fresh_{freq}Hz",
                    indices[0],
                    f"First spike at {freq}Hz (fresh neuron)",
                )
            if len(indices) >= 2:
                add_case(
                    f"A_fatigued_{freq}Hz",
                    indices[-1],
                    f"Last spike at {freq}Hz (fatigued)",
                )

    # --- Protocol B: Refractory recovery curve ---
    refractory_indices = [
        i for i, s in enumerate(spikes) if s["protocol"] == "B_refractory"
    ]
    if refractory_indices:
        # Sort by ISI
        sorted_refr = sorted(
            refractory_indices, key=lambda i: spikes[i].get("isi_ms", 0)
        )
        if len(sorted_refr) >= 3:
            # Short ISI (partial recovery)
            idx = sorted_refr[0]
            add_case(
                "B_short_ISI",
                idx,
                f"Short ISI={spikes[idx]['isi_ms']:.1f}ms (partial recovery)",
            )
            # Medium ISI
            mid_idx = sorted_refr[len(sorted_refr) // 2]
            add_case(
                "B_medium_ISI", mid_idx, f"Medium ISI={spikes[mid_idx]['isi_ms']:.1f}ms"
            )
            # Long ISI (full recovery)
            idx = sorted_refr[-1]
            add_case(
                "B_long_ISI",
                idx,
                f"Long ISI={spikes[idx]['isi_ms']:.1f}ms (full recovery)",
            )

    # --- Protocol C: Population extremes ---
    pop_indices = [i for i, s in enumerate(spikes) if s["protocol"] == "C_population"]
    if pop_indices:
        # Find extreme g_Na/g_K combinations
        # High g_Na, low g_K (fast rise, slow recovery)
        high_na_low_k = max(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            - spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_high_gNa_low_gK",
            high_na_low_k,
            f"High gNa={spikes[high_na_low_k]['g_na_scale']:.2f}, Low gK={spikes[high_na_low_k]['g_k_scale']:.2f}",
        )

        # Low g_Na, high g_K (slow rise, fast recovery)
        low_na_high_k = min(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            - spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_low_gNa_high_gK",
            low_na_high_k,
            f"Low gNa={spikes[low_na_high_k]['g_na_scale']:.2f}, High gK={spikes[low_na_high_k]['g_k_scale']:.2f}",
        )

        # Both high (large amplitude)
        both_high = max(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            + spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_both_high",
            both_high,
            f"Both high: gNa={spikes[both_high]['g_na_scale']:.2f}, gK={spikes[both_high]['g_k_scale']:.2f}",
        )

        # Both low (small amplitude)
        both_low = min(
            pop_indices,
            key=lambda i: spikes[i].get("g_na_scale", 1)
            + spikes[i].get("g_k_scale", 1),
        )
        add_case(
            "C_both_low",
            both_low,
            f"Both low: gNa={spikes[both_low]['g_na_scale']:.2f}, gK={spikes[both_low]['g_k_scale']:.2f}",
        )

    # --- Protocol D: Hyperpolarization rebound ---
    hyperpol_indices = [
        i for i, s in enumerate(spikes) if s["protocol"] == "D_hyperpol"
    ]
    if hyperpol_indices:
        # Group by hyperpolarization level
        hyper_groups = {}
        for idx in hyperpol_indices:
            v_hyper = spikes[idx].get("v_hyper", -65)
            if v_hyper not in hyper_groups:
                hyper_groups[v_hyper] = []
            hyper_groups[v_hyper].append(idx)

        # Get spike from most hyperpolarized (most super-charged)
        most_hyper = min(hyper_groups.keys())
        if hyper_groups[most_hyper]:
            idx = hyper_groups[most_hyper][0]
            peak_v = spikes[idx].get("peak_voltage", 0)
            add_case(
                "D_super_charged",
                idx,
                f"V_hyper={most_hyper}mV, h={spikes[idx]['h_init']:.2f}, peak={peak_v:.1f}mV",
            )

        # Get spike from least hyperpolarized (baseline comparison)
        least_hyper = max(hyper_groups.keys())
        if hyper_groups[least_hyper] and least_hyper != most_hyper:
            idx = hyper_groups[least_hyper][0]
            peak_v = spikes[idx].get("peak_voltage", 0)
            add_case(
                "D_mild_hyper",
                idx,
                f"V_hyper={least_hyper}mV, h={spikes[idx]['h_init']:.2f}, peak={peak_v:.1f}mV",
            )

    # --- Worst-case spike ---
    worst_idx = int(np.argmax(errors_2pc))
    add_case(
        "Worst_case",
        worst_idx,
        f"Highest reconstruction error (RMSE={errors_2pc[worst_idx]:.2f}mV)",
    )

    # --- Best-case spike ---
    best_idx = int(np.argmin(errors_2pc))
    add_case(
        "Best_case",
        best_idx,
        f"Lowest reconstruction error (RMSE={errors_2pc[best_idx]:.2f}mV)",
    )

    # Summary statistics
    validation["summary"] = {
        "n_test_cases": len(validation["test_cases"]),
        "mean_rmse_2pc": float(
            np.mean([tc["rmse_2pc"] for tc in validation["test_cases"]])
        ),
        "max_rmse_2pc": float(
            np.max([tc["rmse_2pc"] for tc in validation["test_cases"]])
        ),
        "mean_rmse_3pc": float(
            np.mean([tc["rmse_3pc"] for tc in validation["test_cases"]])
        ),
        "all_spikes_mean_rmse_2pc": float(np.mean(errors_2pc)),
        "all_spikes_max_rmse_2pc": float(np.max(errors_2pc)),
        "all_spikes_percentile_95_rmse_2pc": float(np.percentile(errors_2pc, 95)),
        "all_spikes_percentile_99_rmse_2pc": float(np.percentile(errors_2pc, 99)),
    }

    return validation


def plot_validation(
    spikes: list[dict],
    pca_result: dict,
    validation: dict,
    output_path: str,
):
    """
    Create multi-panel validation figure showing reconstructions of diverse spike types.

    Layout: 4x3 grid showing 12 test cases with actual vs reconstructed waveforms.
    """
    import matplotlib.pyplot as plt

    test_cases = validation["test_cases"]
    n_cases = len(test_cases)

    # Determine grid size
    n_cols = 3
    n_rows = (n_cases + n_cols - 1) // n_cols
    n_rows = max(n_rows, 4)  # At least 4 rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    for i, tc in enumerate(test_cases):
        ax = axes[i]
        idx = tc["idx"]

        actual = spikes[idx]["waveform"]
        recon_2pc = reconstruct_spike(pca_result, idx, n_components=2)
        recon_3pc = reconstruct_spike(pca_result, idx, n_components=3)

        # Plot
        ax.plot(t_ms, actual, "k-", linewidth=2, label="Actual")
        ax.plot(
            t_ms,
            pca_result["mean"],
            "gray",
            linewidth=1,
            linestyle=":",
            alpha=0.5,
            label="Mean",
        )
        ax.plot(
            t_ms,
            recon_2pc,
            "r--",
            linewidth=1.5,
            label=f"2PC (RMSE={tc['rmse_2pc']:.2f})",
        )
        ax.plot(
            t_ms,
            recon_3pc,
            "b:",
            linewidth=1.5,
            label=f"3PC (RMSE={tc['rmse_3pc']:.2f})",
        )

        ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"{tc['name']}\n{tc['description']}", fontsize=9)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("V (mV)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

        # Color code by quality
        if tc["rmse_2pc"] < 0.3:
            ax.set_facecolor("#e6ffe6")  # Light green - excellent
        elif tc["rmse_2pc"] < 0.5:
            ax.set_facecolor("#ffffcc")  # Light yellow - good
        else:
            ax.set_facecolor("#ffe6e6")  # Light red - needs attention

    # Hide unused axes
    for i in range(n_cases, len(axes)):
        axes[i].axis("off")

    # Add summary text in last panel if space
    if n_cases < len(axes):
        ax_summary = axes[n_cases]
        ax_summary.axis("off")
        summary = validation["summary"]
        summary_text = (
            f"VALIDATION SUMMARY\n"
            f"{'=' * 30}\n\n"
            f"Test Cases: {summary['n_test_cases']}\n\n"
            f"Selected Test Cases:\n"
            f"  Mean RMSE (2PC): {summary['mean_rmse_2pc']:.3f} mV\n"
            f"  Max RMSE (2PC):  {summary['max_rmse_2pc']:.3f} mV\n\n"
            f"All {len(spikes)} Spikes:\n"
            f"  Mean RMSE (2PC): {summary['all_spikes_mean_rmse_2pc']:.3f} mV\n"
            f"  Max RMSE (2PC):  {summary['all_spikes_max_rmse_2pc']:.3f} mV\n"
            f"  95th %ile:       {summary['all_spikes_percentile_95_rmse_2pc']:.3f} mV\n"
            f"  99th %ile:       {summary['all_spikes_percentile_99_rmse_2pc']:.3f} mV\n\n"
            f"Color Legend:\n"
            f"  Green:  RMSE < 0.3 mV (excellent)\n"
            f"  Yellow: RMSE < 0.5 mV (good)\n"
            f"  Red:    RMSE >= 0.5 mV (review)"
        )
        ax_summary.text(
            0.1,
            0.9,
            summary_text,
            transform=ax_summary.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
        )

    plt.suptitle(
        "Basis Function Validation: Reconstruction Quality Across Spike Types",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=150)
    print(f"Saved validation plot to: {output_path}")
    plt.close(fig)


# =============================================================================
# Visualization
# =============================================================================


def plot_results(spikes: list[dict], pca_result: dict, output_path: str):
    """Create 3-panel figure showing spike library, eigen-spikes, and reconstruction."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Time axis for plotting (in ms, relative to peak)
    t_ms = np.arange(WINDOW_POINTS) * DT_MS - WINDOW_PRE_MS

    # -------------------------------------------------------------------------
    # Panel 1: All aligned spikes + Mean
    # -------------------------------------------------------------------------
    ax1 = axes[0]

    # Color by protocol
    colors = {
        "A_fatigue": "red",
        "B_refractory": "blue",
        "C_population": "gray",
        "D_hyperpol": "green",
    }

    # Plot all spikes with low alpha
    for s in spikes:
        color = colors.get(s["protocol"], "gray")
        alpha = 0.3 if s["protocol"] == "C_population" else 0.5
        ax1.plot(t_ms, s["waveform"], color=color, alpha=alpha, linewidth=0.5)

    # Plot mean
    ax1.plot(t_ms, pca_result["mean"], "k-", linewidth=2.5, label="Mean")

    # Add legend entries for protocols
    ax1.plot(
        [],
        [],
        "r-",
        linewidth=2,
        label=f"Fatigue (n={sum(1 for s in spikes if s['protocol'] == 'A_fatigue')})",
    )
    ax1.plot(
        [],
        [],
        "b-",
        linewidth=2,
        label=f"Refractory (n={sum(1 for s in spikes if s['protocol'] == 'B_refractory')})",
    )
    ax1.plot(
        [],
        [],
        "gray",
        linewidth=2,
        label=f"Population (n={sum(1 for s in spikes if s['protocol'] == 'C_population')})",
    )
    ax1.plot(
        [],
        [],
        "g-",
        linewidth=2,
        label=f"Hyperpol (n={sum(1 for s in spikes if s['protocol'] == 'D_hyperpol')})",
    )

    ax1.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.axhline(-65, color="gray", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Time relative to peak (ms)")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_title(f"Spike Library: {len(spikes)} aligned spikes")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    # -------------------------------------------------------------------------
    # Panel 2: Eigen-spikes (PC1, PC2, PC3)
    # -------------------------------------------------------------------------
    ax2 = axes[1]

    components = pca_result["components"]
    var_ratios = pca_result["explained_variance_ratio"]

    # Scale components for visualization (they're unit vectors)
    scale = 20.0  # Scale factor for visibility

    ax2.plot(
        t_ms,
        components[0] * scale,
        "r-",
        linewidth=2,
        label=f"PC1 ({var_ratios[0] * 100:.1f}%) - Width/Duration",
    )
    ax2.plot(
        t_ms,
        components[1] * scale,
        "b-",
        linewidth=2,
        label=f"PC2 ({var_ratios[1] * 100:.1f}%) - AHP Depth",
    )
    if len(components) > 2:
        ax2.plot(
            t_ms,
            components[2] * scale,
            "g-",
            linewidth=2,
            label=f"PC3 ({var_ratios[2] * 100:.1f}%) - Rise/Fall Asymmetry",
        )

    ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Time relative to peak (ms)")
    ax2.set_ylabel(f"PC amplitude (scaled by {scale})")
    ax2.set_title("Eigen-Spikes: Principal Components")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    # -------------------------------------------------------------------------
    # Panel 3: Reconstruction of "Most Fatigued" spike
    # -------------------------------------------------------------------------
    ax3 = axes[2]

    # Find the most fatigued spike (last spike from Protocol A)
    fatigue_spikes = [i for i, s in enumerate(spikes) if s["protocol"] == "A_fatigue"]
    if fatigue_spikes:
        # Use the last fatigue spike (most fatigued)
        target_idx = fatigue_spikes[-1]
        target_spike = spikes[target_idx]

        actual = target_spike["waveform"]
        recon_2pc = reconstruct_spike(pca_result, target_idx, n_components=2)
        recon_3pc = reconstruct_spike(pca_result, target_idx, n_components=3)

        # Calculate RMSE
        rmse_2pc = np.sqrt(np.mean((actual - recon_2pc) ** 2))
        rmse_3pc = np.sqrt(np.mean((actual - recon_3pc) ** 2))

        ax3.plot(t_ms, actual, "k-", linewidth=2.5, label="Actual (Most Fatigued)")
        ax3.plot(
            t_ms,
            pca_result["mean"],
            "gray",
            linewidth=1.5,
            linestyle="--",
            label="Mean only",
        )
        ax3.plot(
            t_ms,
            recon_2pc,
            "r--",
            linewidth=2,
            label=f"Mean + 2 PCs (RMSE={rmse_2pc:.2f} mV)",
        )
        ax3.plot(
            t_ms,
            recon_3pc,
            "b:",
            linewidth=2,
            label=f"Mean + 3 PCs (RMSE={rmse_3pc:.2f} mV)",
        )

        # Show weights used
        weights = pca_result["weights"][target_idx]
        weight_text = f"Weights: w1={weights[0]:.1f}, w2={weights[1]:.1f}"
        if len(weights) > 2:
            weight_text += f", w3={weights[2]:.1f}"
        ax3.text(
            0.02,
            0.98,
            weight_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        title = f"Reconstruction: Spike #{target_spike['spike_num']} from Fatigue Train"
    else:
        # Fallback: use first spike
        actual = spikes[0]["waveform"]
        recon_2pc = reconstruct_spike(pca_result, 0, n_components=2)
        ax3.plot(t_ms, actual, "k-", linewidth=2.5, label="Actual")
        ax3.plot(t_ms, recon_2pc, "r--", linewidth=2, label="Mean + 2 PCs")
        title = "Reconstruction: First Spike"

    ax3.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax3.set_xlabel("Time relative to peak (ms)")
    ax3.set_ylabel("Voltage (mV)")
    ax3.set_title(title)
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-WINDOW_PRE_MS, WINDOW_POST_MS)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("SPIKE SHAPE DECOMPOSITION SYSTEM (SSDS)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Time step: {DT_MS} ms")
    print(
        f"  Window: -{WINDOW_PRE_MS}ms to +{WINDOW_POST_MS}ms ({WINDOW_POINTS} points)"
    )
    print(f"  Spike threshold: {SPIKE_THRESHOLD_MV} mV")
    print()

    # Generate spike library
    print("-" * 70)
    print("GENERATING SPIKE LIBRARY")
    print("-" * 70)

    spikes_a = generate_protocol_a()
    spikes_b = generate_protocol_b()
    spikes_c = generate_protocol_c(n_samples=1000)
    spikes_d = generate_protocol_d(n_samples=50)

    all_spikes = spikes_a + spikes_b + spikes_c + spikes_d
    print(f"\nTotal spikes collected: {len(all_spikes)}")

    # Run PCA
    print("-" * 70)
    print("PCA ANALYSIS")
    print("-" * 70)

    pca_result = run_pca_analysis(all_spikes, n_components=3)

    # Compute reconstruction errors for all spikes
    print("\nReconstruction Error Analysis:")
    X = pca_result["X"]
    mean = pca_result["mean"]

    # Mean-only error
    mean_only_error = np.sqrt(np.mean((X - mean) ** 2, axis=1))
    print(
        f"  Mean only:  RMSE = {np.mean(mean_only_error):.2f} ± {np.std(mean_only_error):.2f} mV"
    )

    # 2-PC reconstruction error
    recon_2pc = mean + pca_result["weights"][:, :2] @ pca_result["components"][:2]
    error_2pc = np.sqrt(np.mean((X - recon_2pc) ** 2, axis=1))
    print(f"  Mean + 2PC: RMSE = {np.mean(error_2pc):.2f} ± {np.std(error_2pc):.2f} mV")

    # 3-PC reconstruction error
    recon_3pc = mean + pca_result["weights"][:, :3] @ pca_result["components"][:3]
    error_3pc = np.sqrt(np.mean((X - recon_3pc) ** 2, axis=1))
    print(f"  Mean + 3PC: RMSE = {np.mean(error_3pc):.2f} ± {np.std(error_3pc):.2f} mV")

    # Visualize
    print("-" * 70)
    print("VISUALIZATION")
    print("-" * 70)

    plot_results(all_spikes, pca_result, "spike_pca_analysis.png")

    # Validation
    print("-" * 70)
    print("RECONSTRUCTION VALIDATION")
    print("-" * 70)

    validation = validate_reconstructions(all_spikes, pca_result)

    print(f"\nValidation Test Cases ({validation['summary']['n_test_cases']} spikes):")
    for tc in validation["test_cases"]:
        status = "OK" if tc["rmse_2pc"] < 0.5 else "REVIEW"
        print(
            f"  [{status}] {tc['name']}: RMSE={tc['rmse_2pc']:.3f}mV - {tc['description']}"
        )

    print(f"\nValidation Summary:")
    summary = validation["summary"]
    print(f"  Test cases mean RMSE (2PC): {summary['mean_rmse_2pc']:.3f} mV")
    print(f"  Test cases max RMSE (2PC):  {summary['max_rmse_2pc']:.3f} mV")
    print(
        f"  All spikes 95th percentile: {summary['all_spikes_percentile_95_rmse_2pc']:.3f} mV"
    )
    print(
        f"  All spikes 99th percentile: {summary['all_spikes_percentile_99_rmse_2pc']:.3f} mV"
    )

    plot_validation(all_spikes, pca_result, validation, "spike_validation.png")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_var = sum(pca_result["explained_variance_ratio"][:2]) * 100
    print(f"  2 PCs capture {total_var:.1f}% of variance")
    print(f"  Mean reconstruction error with 2 PCs: {np.mean(error_2pc):.2f} mV")
    print(f"  This proves we can represent complex spike shapes with:")
    print(f"    - 1 static Mean waveform ({WINDOW_POINTS} floats)")
    print(f"    - 2 PC basis vectors ({2 * WINDOW_POINTS} floats)")
    print(f"    - 2 dynamic weights per spike (2 floats)")
    print("=" * 70)

    # =========================================================================
    # Export Basis Functions
    # =========================================================================
    print("\n" + "-" * 70)
    print("EXPORTING BASIS FUNCTIONS")
    print("-" * 70)

    # Extract top 3 components (shape: 3 x N where N = WINDOW_POINTS)
    components_3 = pca_result["components"][:3]  # Shape: (3, 1000)
    explained_variance_3 = pca_result["explained_variance_ratio"][:3]

    # Save to npz file
    output_file = "basis_data.npz"
    np.savez(
        output_file,
        mean_waveform=pca_result["mean"],  # Shape: (1000,)
        components=components_3,  # Shape: (3, 1000)
        explained_variance=explained_variance_3,  # Shape: (3,)
        dt_ms=DT_MS,
        window_pre_ms=WINDOW_PRE_MS,
        window_post_ms=WINDOW_POST_MS,
        window_samples=WINDOW_POINTS,
    )

    print(f"  Saved basis functions to: {output_file}")
    print(f"  Contents:")
    print(f"    - mean_waveform: shape {pca_result['mean'].shape}")
    print(f"    - components: shape {components_3.shape}")
    print(f"    - explained_variance: {explained_variance_3}")
    print(f"    - dt_ms: {DT_MS}")
    print(f"    - window_pre_ms: {WINDOW_PRE_MS}")
    print(f"    - window_post_ms: {WINDOW_POST_MS}")
    print(f"    - window_samples: {WINDOW_POINTS}")
    print("=" * 70)


if __name__ == "__main__":
    main()
