
import numpy as np
from .physics import HHPhysics

DT_MS = 0.01  # Default Time step (ms)
STIM_DURATION_MS = 1.0  # Default duration of each stimulus pulse
STIM_AMPLITUDE = 30.0   # Default amplitude

def create_pulse_train(
    t_end_ms: float,
    pulse_times_ms: list[float],
    dt_ms: float = DT_MS,
    stim_duration_ms: float = STIM_DURATION_MS,
    stim_amplitude: float = STIM_AMPLITUDE
) -> np.ndarray:
    """Create stimulus array with pulses at specified times."""
    n_time = int(t_end_ms / dt_ms)
    i_stim = np.zeros(n_time)

    for t_pulse in pulse_times_ms:
        start_idx = int(t_pulse / dt_ms)
        end_idx = int((t_pulse + stim_duration_ms) / dt_ms)
        end_idx = min(end_idx, n_time)
        if start_idx < n_time:
            i_stim[start_idx:end_idx] = stim_amplitude

    return i_stim

def run_simulation(
    t_end_ms: float,
    i_stim: np.ndarray,
    *,
    dt_ms: float = DT_MS,
    g_na_scale: float = 1.0,
    g_k_scale: float = 1.0,
    v_init: float | None = None,
    gates_init: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run HH simulation with custom stimulus and optional initial conditions.

    Parameters
    ----------
    t_end_ms : float
        Simulation duration in ms.
    i_stim : np.ndarray
        Stimulus current array (length must match n_time).
    dt_ms : float
        Time step in ms.
    g_na_scale : float
        Scale factor for g_Na (1.0 = default).
    g_k_scale : float
        Scale factor for g_K (1.0 = default).
    v_init : float | None
        Initial voltage. If None, uses v_rest.
    gates_init : tuple[float, float, float] | None
        Initial (m, h, n) values. If None, uses steady-state at v_init.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (t_ms, V) arrays.
    """
    n_time = int(t_end_ms / dt_ms)
    t_ms = np.arange(n_time) * dt_ms

    if len(i_stim) != n_time:
        raise ValueError(f"i_stim length ({len(i_stim)}) != n_time ({n_time})")

    # Conductances
    g_Na = HHPhysics.g_Na * g_na_scale
    g_K = HHPhysics.g_K * g_k_scale
    g_L = HHPhysics.g_L
    E_Na, E_K, E_L = HHPhysics.E_Na, HHPhysics.E_K, HHPhysics.E_L
    inv_C_m = 1.0 / HHPhysics.C_m

    # Initial conditions
    V = v_init if v_init is not None else HHPhysics.v_rest

    if gates_init is not None:
        m, h, n = gates_init
    else:
        m, h, n = HHPhysics.steady_state(V)

    # History
    V_hist = np.zeros(n_time)
    V_hist[0] = V

    # Integration loop
    for i in range(1, n_time):
        # Ionic currents
        I_Na = g_Na * (m**3) * h * (V - E_Na)
        I_K = g_K * (n**4) * (V - E_K)
        I_L = g_L * (V - E_L)

        # Gate kinetics (using safe rates)
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = HHPhysics.rates(V)

        dm = alpha_m * (1.0 - m) - beta_m * m
        dh = alpha_h * (1.0 - h) - beta_h * h
        dn = alpha_n * (1.0 - n) - beta_n * n

        m += dm * dt_ms
        h += dh * dt_ms
        n += dn * dt_ms

        # Voltage update
        dV = (i_stim[i - 1] - I_Na - I_K - I_L) * inv_C_m
        V += dV * dt_ms

        V_hist[i] = V

    return t_ms, V_hist
