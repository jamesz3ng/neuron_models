import matplotlib.pyplot as plt
import numpy as np

# https://goldmanlab.faculty.ucdavis.edu/wp-content/uploads/sites/263/2016/07/HodgkinHuxley.pdf
# values taken from above converted to SI units

# constants
C_m = 1.0  # nF/mm^2
g_Na = 120.0  # uS/cm^2
g_K = 36.0  # uS/cm^2
g_L = 0.3  # uS/cm^2

# reverse potential (mV)
E_Na = 50.0
E_K = -77.0
E_L = -54.387

# cabel geometry

length = 2.0  # cm
diameter = 0.05  #
radius = diameter / 2
r_a = 35.4  # axial resistivity

# discretization
n_spatial = 100
dx = length / n_spatial
dt = 0.01
T = 50.0

diffusion_coeff = radius / (2 * r_a * dx**2 + C_m)

# rate functions
# k activation


def alpha_n(V):
    return -0.01 * (V + 55) / (np.exp((V + 55) / -10) - 1)


def beta_n(V):
    return 0.125 * np.exp((V + 65) / -80)


def alpha_m(V):
    return -0.1 * (V + 40) / (np.exp((V + 40) / -10) - 1)


def beta_m(V):
    return 4 * np.exp((V + 65) / -18)


def alpha_h(V):
    return 0.07 * np.exp((V + 65) / -20)


def beta_h(V):
    return 1 / (1 + np.exp((V + 35) / -10))


def create_square_pulse(t, start, end, amplitude, dt):
    pulse = np.zeros_like(t)
    pulse[int(start / dt) : int(end / dt)] = amplitude
    return pulse


def main():
    n_time = int(T / dt)
    t = np.arange(0, T, dt)
    x = np.linspace(0, length, n_spatial)

    V = np.zeros((n_time, n_spatial))
    m = np.zeros((n_time, n_spatial))
    h = np.zeros((n_time, n_spatial))
    n = np.zeros((n_time, n_spatial))

    # set initial resting positions
    V[0, :] = -65.0
    m[0, :] = 0.0529
    h[0, :] = 0.5961
    n[0, :] = 0.3177

    # I_inj = create_square_pulse(t, 20, 60, 20, dt)
    I_inj = np.zeros((n_time, n_spatial))
    start_idx = int(20 / dt)
    end_idx = int(30 / dt)
    I_inj[start_idx:end_idx, 0] = 20.0

    # Using the Euler forward method
    for i in range(1, len(t)):
        V_old = V[i - 1, :]
        m_old = m[i - 1, :]
        h_old = h[i - 1, :]
        n_old = n[i - 1, :]

        I_Na = g_Na * (m_old**3) * h_old * (V_old - E_Na)
        I_K = g_K * (n_old**4) * (V_old - E_K)
        I_L = g_L * (V_old - E_L)

        dm = alpha_m(V_old) * (1 - m_old) - beta_m(V_old) * m_old
        dh = alpha_h(V_old) * (1 - h_old) - beta_h(V_old) * h_old
        dn = alpha_n(V_old) * (1 - n_old) - beta_n(V_old) * n_old

        m[i, :] = m_old + dm * dt
        h[i, :] = h_old + dh * dt
        n[i, :] = n_old + dn * dt

        # cable equation: spatial diffusion term
        d2V = np.zeros(n_spatial)
        d2V[1:-1] = V_old[2:] - 2 * V_old[1:-1] + V_old[:-2]
        d2V[0] = V_old[1] - V_old[0]
        d2V[-1] = V_old[-2] - V_old[-1]

        dV = (diffusion_coeff * d2V + I_inj[i - 1, :] - I_Na - I_K - I_L) / C_m

        V[i, :] = V_old + dV * dt

    # 1. Space-time plot (heatmap)
    ax1 = plt.subplot(3, 1, 1)
    im = ax1.imshow(
        V.T,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-80,
        vmax=40,
    )
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Position (cm)")
    ax1.set_ylim(0, length)
    ax1.set_title("Action Potential Propagation Along Cable")
    plt.colorbar(im, ax=ax1, label="Membrane Potential (mV)")

    # 2. Voltage at different positions
    ax2 = plt.subplot(3, 1, 2)
    positions_to_plot = [
        0,
        n_spatial // 4,
        n_spatial // 2,
        3 * n_spatial // 4,
        n_spatial - 1,
    ]
    for pos in positions_to_plot:
        ax2.plot(t, V[:, pos], label=f"x = {x[pos]:.2f} cm")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Membrane Potential (mV)")
    ax2.set_title("Voltage at Different Positions")
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(3, 1, 3)
    times_to_plot = [10, 20, 25, 30, 35]
    for time_ms in times_to_plot:
        idx = int(time_ms / dt)
        if idx < n_time:
            ax3.plot(x, V[idx, :], label=f"t = {time_ms} ms")
    ax3.set_xlabel("Position (cm)")
    ax3.set_ylabel("Membrane Potential (mV)")
    ax3.set_title("Spatial Profile at Different Times")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig("hh_cable_model.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
