import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def main():
    """Main function for wave model simulation."""
    # constants
    L = 1.0
    c = 10.0  # conduction velocity for an myelinated axon is between 10 - 150m/s
    T = 0.5

    nx = 100  # number of spatial points
    dx = L / (nx - 1)


    dt = 1 * dx / c
    nt = int(T / dt)

    alpha = (c * dt / dx) ** 2

    # initialisation
    x = np.linspace(0, L, nx)
    v_prev = np.zeros(nx)
    v_curr = np.zeros(nx)
    v_next = np.zeros(nx)

    # create initial spike at x=0.2 using the gaussian function
    # one wave
    v_curr = np.exp(-100 * (x - 0.2) ** 2)
    v_prev = np.exp(-100 * (x - 0.2 + c * dt) ** 2)

    history = []
    times = []

    # plot where the cancellations happen

    for n in range(nt):
        v_next[1:-1] = (
            2 * v_curr[1:-1]
            - v_prev[1:-1]
            + alpha * (v_curr[2:] - 2 * v_curr[1:-1] + v_curr[:-2])
        )

        # v_next[-1] = 0

        # absorbing boundary condition the wave passes through
        v_next[-1] = v_curr[-2]
        v_next[0] = 0

        v_prev[:] = v_curr[:]
        v_curr[:] = v_next[:]

        if n % 6 == 0:
            history.append(v_curr.copy())
            times.append(n * dt)

    # --- PLOTTING ---
    # We will create an Animation so you see ONE wave moving, not a history trail
    fig, ax = plt.subplots(figsize=(10, 5))
    (line,) = ax.plot([], [], lw=2, color="blue")

    ax.set_xlim(0, L)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title("Single Action Potential Propagation")
    ax.set_xlabel("Axon Length (x)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)


    def init():
        line.set_data([], [])
        return (line,)


    def animate(i):
        # Only take every 2nd captured frame to speed up animation
        idx = i
        if idx < len(history):
            y = history[idx]
            line.set_data(x, y)
            ax.set_title(f"Time: {times[idx]}s")
        return (line,)


    anim = FuncAnimation(
        fig, animate, init_func=init, frames=len(history), interval=30, blit=False
    )

    plt.show()


if __name__ == "__main__":
    main()
