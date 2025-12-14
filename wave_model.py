import numpy as np


def _default_params():
    return {
        "L": 1.0,
        "c": 10.0,
        "T": 0.5,
        "nx": 100,
        "dt": None,  # if None, uses dx / c
        "spike_center": 0.2,
        "spike_width": 100.0,
        "store_history": True,
        "history_stride": 6,
    }


def simulate_wave_model(
    *,
    L: float | None = None,
    c: float | None = None,
    T: float | None = None,
    nx: int | None = None,
    dt: float | None = None,
    spike_center: float | None = None,
    spike_width: float | None = None,
    store_history: bool | None = None,
    history_stride: int | None = None,
):
    defaults = _default_params()
    L = defaults["L"] if L is None else L
    c = defaults["c"] if c is None else c
    T = defaults["T"] if T is None else T
    nx = defaults["nx"] if nx is None else nx
    dt = defaults["dt"] if dt is None else dt
    spike_center = defaults["spike_center"] if spike_center is None else spike_center
    spike_width = defaults["spike_width"] if spike_width is None else spike_width
    store_history = defaults["store_history"] if store_history is None else store_history
    history_stride = (
        defaults["history_stride"] if history_stride is None else history_stride
    )

    dx = L / (nx - 1)
    dt = dx / c if dt is None else dt
    nt = int(T / dt)

    alpha = (c * dt / dx) ** 2

    x = np.linspace(0, L, nx)
    v_prev = np.zeros(nx)
    v_curr = np.zeros(nx)
    v_next = np.zeros(nx)

    v_curr = np.exp(-spike_width * (x - spike_center) ** 2)
    v_prev = np.exp(-spike_width * (x - (spike_center + c * dt)) ** 2)

    history: list[np.ndarray] = []
    times: list[float] = []

    for n in range(nt):
        v_next[1:-1] = (
            2 * v_curr[1:-1]
            - v_prev[1:-1]
            + alpha * (v_curr[2:] - 2 * v_curr[1:-1] + v_curr[:-2])
        )

        v_next[-1] = v_curr[-2]
        v_next[0] = 0

        v_prev[:] = v_curr[:]
        v_curr[:] = v_next[:]

        if store_history and (n % history_stride == 0):
            history.append(v_curr.copy())
            times.append(n * dt)

    result = {
        "x": x,
        "alpha": float(alpha),
        "dt": float(dt),
        "T": float(T),
        "L": float(L),
        "nx": int(nx),
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
