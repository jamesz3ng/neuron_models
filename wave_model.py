import numpy as np


def _default_params():
    """
    Unit conventions:
    - Time is in seconds (s).
    - Length is in arbitrary units consistent with `c` (typically meters if `c` is m/s).
    """
    return {
        "L": 1.0,
        "c": 10.0,
        "T_s": 0.5,
        "nx": 100,
        "dt_s": None,  # if None, uses dx / c
        "spike_center": 0.2,
        "spike_width": 100.0,
        "store_history": True,
        "history_stride": 6,
    }


def simulate_wave_model(
    *,
    L: float | None = None,
    c: float | None = None,
    T_s: float | None = None,
    nx: int | None = None,
    dt_s: float | None = None,
    spike_center: float | None = None,
    spike_width: float | None = None,
    store_history: bool | None = None,
    history_stride: int | None = None,
    v_input: np.ndarray | None = None,
    t_input: np.ndarray | None = None,
    v_init: float | None = None,
):
    """
    Simulate 1D wave equation for action potential propagation.

    Can be initialized either with:
    1. Gaussian pulse (spike_center, spike_width) - creates two counter-propagating waves
    2. Input waveform (v_input, t_input) - injects waveform at x=0 boundary

    Parameters
    ----------
    L : float
        Cable length
    c : float
        Wave propagation velocity
    T_s : float
        Simulation duration in seconds
    nx : int
        Number of spatial points
    dt_s : float
        Time step (if None, uses CFL condition dt = dx/c)
    spike_center : float
        Center position for Gaussian initial condition (ignored if v_input provided)
    spike_width : float
        Width parameter for Gaussian (higher = narrower)
    store_history : bool
        Whether to store full history
    history_stride : int
        Stride for history storage
    v_input : np.ndarray
        Input voltage waveform to inject at x=0 (optional)
    t_input : np.ndarray
        Time array for v_input in seconds (required if v_input provided)
    v_init : float
        Initial value for entire grid (used with v_input for proper initialization)
    """
    defaults = _default_params()
    L = defaults["L"] if L is None else L
    c = defaults["c"] if c is None else c
    T_s = defaults["T_s"] if T_s is None else T_s
    nx = defaults["nx"] if nx is None else nx
    dt_s = defaults["dt_s"] if dt_s is None else dt_s
    spike_center = defaults["spike_center"] if spike_center is None else spike_center
    spike_width = defaults["spike_width"] if spike_width is None else spike_width
    store_history = (
        defaults["store_history"] if store_history is None else store_history
    )
    history_stride = (
        defaults["history_stride"] if history_stride is None else history_stride
    )

    dx = L / (nx - 1)
    dt_s = dx / c if dt_s is None else dt_s
    nt = int(T_s / dt_s)

    alpha = (c * dt_s / dx) ** 2

    x = np.linspace(0, L, nx)

    # Prepare boundary input if provided
    v_boundary: np.ndarray | None = None
    if v_input is not None and t_input is not None:
        # Interpolate input waveform to simulation time grid
        t_sim = np.arange(nt) * dt_s
        v_boundary = np.interp(
            t_sim, t_input, v_input, left=v_input[0], right=v_input[-1]
        )
        use_input_boundary = True

        # Initialize grid to v_init (or first input value if not specified)
        init_val = v_init if v_init is not None else v_input[0]
        v_prev = np.full(nx, init_val)
        v_curr = np.full(nx, init_val)
        v_next = np.full(nx, init_val)
    else:
        use_input_boundary = False
        # Use Gaussian initial condition
        v_prev = np.zeros(nx)
        v_curr = np.zeros(nx)
        v_next = np.zeros(nx)
        v_curr = np.exp(-spike_width * (x - spike_center) ** 2)
        v_prev = np.exp(-spike_width * (x - (spike_center + c * dt_s)) ** 2)

    history: list[np.ndarray] = []
    times: list[float] = []

    for n in range(nt):
        # Wave equation update for interior points
        v_next[1:-1] = (
            2 * v_curr[1:-1]
            - v_prev[1:-1]
            + alpha * (v_curr[2:] - 2 * v_curr[1:-1] + v_curr[:-2])
        )

        # Right boundary: Neumann (zero flux / non-reflecting)
        v_next[-1] = v_curr[-2]

        # Left boundary: either inject input or absorbing
        if use_input_boundary and v_boundary is not None:
            v_next[0] = v_boundary[n]
        else:
            v_next[0] = 0  # Absorbing boundary

        v_prev[:] = v_curr[:]
        v_curr[:] = v_next[:]

        if store_history and (n % history_stride == 0):
            history.append(v_curr.copy())
            times.append(n * dt_s)

    result = {
        "x": x,
        "alpha": float(alpha),
        "dt_s": float(dt_s),
        "T_s": float(T_s),
        "L": float(L),
        "nx": int(nx),
        "n_t": int(nt),
    }
    if store_history:
        result["history"] = history
        result["times"] = times
    else:
        result["v_final"] = v_curr.copy()
    return result


def main():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    result = simulate_wave_model(store_history=True, history_stride=6)
    x = result["x"]
    history = result["history"]
    times = result["times"]

    fig, ax = plt.subplots(figsize=(10, 5))
    (line,) = ax.plot([], [], lw=2, color="blue")

    ax.set_xlim(0, result["L"])
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Single Action Potential Propagation")
    ax.set_xlabel("Axon Length (x)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        if i < len(history):
            line.set_data(x, history[i])
            ax.set_title(f"Time: {times[i]}s")
        return (line,)

    _anim = FuncAnimation(
        fig, animate, init_func=init, frames=len(history), interval=30, blit=False
    )
    plt.show()


if __name__ == "__main__":
    main()
