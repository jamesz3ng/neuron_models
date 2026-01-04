"""
Passive Cable Equation Model

Simulates voltage spread along a passive cable (dendrite/axon) with:
- Diffusion (spatial spread based on length constant λ)
- Decay (temporal decay based on time constant τ)

No active ion channels - just passive propagation like a "leaky wire".
"""

import numpy as np


def _default_params():
    """
    Unit conventions for the public API:
    - Time is in milliseconds (ms).
    - Length is in micrometers (μm).
    """
    return {
        "L": 1000.0,  # Cable length (μm)
        "T_ms": 40.0,  # Simulation duration (ms)
        "dx": 40.0,  # Spatial step (μm)
        "dt_ms": 0.05,  # Time step (ms)
        "tau_ms": 10.0,  # Membrane time constant (ms)
        "lam": 200.0,  # Length constant (μm)
        "v_rest": 0.0,  # Resting potential (mV, relative)
    }


def simulate_cable_equation(
    *,
    L: float | None = None,
    T_ms: float | None = None,
    dx: float | None = None,
    dt_ms: float | None = None,
    tau_ms: float | None = None,
    lam: float | None = None,
    v_rest: float | None = None,
    v_input: np.ndarray | None = None,
    t_ms_input: np.ndarray | None = None,
    store_history: bool = True,
    history_stride: int = 1,
) -> dict:
    """
    Simulate passive cable equation with optional input waveform injection at x=0.

    Parameters
    ----------
    L : float
        Cable length in μm.
    T_ms : float
        Simulation duration in ms.
    dx : float
        Spatial step in μm.
    dt_ms : float
        Time step in ms.
    tau_ms : float
        Membrane time constant in ms.
    lam : float
        Length constant in μm.
    v_rest : float
        Resting potential in mV.
    v_input : np.ndarray, optional
        Input voltage waveform to inject at x=0. If None, uses a simple pulse.
    t_ms_input : np.ndarray, optional
        Time array for v_input. Must be provided if v_input is provided.
    store_history : bool
        Whether to store full voltage history.
    history_stride : int
        Stride for history storage.

    Returns
    -------
    dict with keys:
        - v_matrix or v_final: voltage data
        - alpha: diffusion stability parameter
        - n_x, n_t: grid dimensions
        - L, T_ms, dx, dt_ms: parameters used
    """
    defaults = _default_params()
    L = defaults["L"] if L is None else L
    T_ms = defaults["T_ms"] if T_ms is None else T_ms
    dx = defaults["dx"] if dx is None else dx
    dt_ms = defaults["dt_ms"] if dt_ms is None else dt_ms
    tau_ms = defaults["tau_ms"] if tau_ms is None else tau_ms
    lam = defaults["lam"] if lam is None else lam
    v_rest = defaults["v_rest"] if v_rest is None else v_rest

    n_x = int(L / dx)
    n_t = int(T_ms / dt_ms)

    # Stability parameters
    alpha = (lam**2 * dt_ms) / (dx**2 * tau_ms)  # Diffusion coefficient
    beta = dt_ms / tau_ms  # Decay coefficient

    # Prepare input waveform (interpolate to simulation time grid if provided)
    if v_input is not None and t_ms_input is not None:
        t_sim = np.arange(n_t) * dt_ms
        v_boundary = np.interp(t_sim, t_ms_input, v_input, left=v_rest, right=v_rest)
    else:
        # Default: simple pulse for first 5ms
        v_boundary = np.full(n_t, v_rest)
        input_steps = int(5.0 / dt_ms)
        v_boundary[:input_steps] = 20.0

    if store_history:
        v_matrix = np.full((n_t, n_x), v_rest, dtype=float)
        v_matrix[0, 0] = v_boundary[0]

        for n in range(n_t - 1):
            v = v_matrix[n, :]

            # Interior points: vectorized diffusion + decay
            # second_derivative[i] = v[i+1] - 2*v[i] + v[i-1]
            second_derivative = v[2:] - 2 * v[1:-1] + v[:-2]
            diffusion = alpha * second_derivative
            decay = beta * (v[1:-1] - v_rest)
            v_matrix[n + 1, 1:-1] = v[1:-1] + diffusion - decay

            # Right boundary: Neumann (zero flux)
            v_matrix[n + 1, -1] = v_matrix[n + 1, -2]

            # Left boundary: inject input waveform
            v_matrix[n + 1, 0] = v_boundary[n + 1]

        return {
            "v_matrix": v_matrix[::history_stride, :],
            "alpha": float(alpha),
            "beta": float(beta),
            "n_x": int(n_x),
            "n_t": int(n_t),
            "L": float(L),
            "T_ms": float(T_ms),
            "dx": float(dx),
            "dt_ms": float(dt_ms),
        }

    # No history storage - just compute final state (vectorized)
    v = np.full(n_x, v_rest, dtype=float)
    v[0] = v_boundary[0]

    for n in range(n_t - 1):
        # Interior points: vectorized diffusion + decay
        second_derivative = v[2:] - 2 * v[1:-1] + v[:-2]
        diffusion = alpha * second_derivative
        decay = beta * (v[1:-1] - v_rest)

        v_new = v.copy()
        v_new[1:-1] = v[1:-1] + diffusion - decay

        # Right boundary: Neumann (zero flux)
        v_new[-1] = v_new[-2]

        # Left boundary: inject input waveform
        v_new[0] = v_boundary[n + 1]

        v = v_new

    return {
        "v_final": v,
        "alpha": float(alpha),
        "beta": float(beta),
        "n_x": int(n_x),
        "n_t": int(n_t),
        "L": float(L),
        "T_ms": float(T_ms),
        "dx": float(dx),
        "dt_ms": float(dt_ms),
    }


def main():
    import matplotlib.pyplot as plt

    # Run with default parameters (simple pulse input)
    result = simulate_cable_equation(store_history=True)
    v_matrix = result["v_matrix"]
    alpha = result["alpha"]
    L = result["L"]
    T_ms = result["T_ms"]

    print(f"Stability alpha = {alpha:.4f} (should be < 0.5)")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        v_matrix,
        aspect="auto",
        cmap="viridis",
        origin="lower",
        extent=(0, L, 0, T_ms),
    )
    plt.colorbar(label="Voltage (mV)")
    plt.xlabel("Distance along cable (μm)")
    plt.ylabel("Time (ms)")
    plt.title("Passive Cable Equation - Voltage Propagation")
    plt.show()


if __name__ == "__main__":
    main()
