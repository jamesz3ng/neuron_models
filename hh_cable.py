import matplotlib.pyplot as plt
import numpy as np

# spatial and temporal parameters
L = 1000.0
T = 40.0
dx = 10.0
dt = 0.01

n_x = int(L / dx)
n_t = int(T / dt)

# cable parameters
lam = 200.0  # space constant
tau = 1.0  # time constant
c_m = 1.0

# hodgkin-huxley parameters
g_Na = 120.0  # uS/cm^2
g_K = 36.0  # uS/cm^2
g_L = 0.3  # uS/cm^2

E_Na = 50.0
E_K = -77.0
E_L = -54.387

# diffusion coefficient
alpha = (lam**2 * dt) / (tau * dx**2)


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


def main():
    v_matrix = np.zeros((n_t, n_x))
    m_matrix = np.zeros((n_t, n_x))
    n_matrix = np.zeros((n_t, n_x))
    h_matrix = np.zeros((n_t, n_x))

    # initial conditions
    v_rest = -65.0
    v_matrix[:, 0] = v_rest

    m_matrix[0, :] = alpha_m(v_rest) / (alpha_m(v_rest) + beta_m(v_rest))
    h_matrix[0, :] = alpha_h(v_rest) / (alpha_h(v_rest) + beta_h(v_rest))
    n_matrix[0, :] = alpha_n(v_rest) / (alpha_n(v_rest) + beta_n(v_rest))

    input_steps = int(5.0 / dt)
    i_stim = 50.0

    # main sim
    for t in range(0, n_t - 1):
        for x in range(1, n_x - 1):
            v = v_matrix[t, x]
            m = m_matrix[t, x]
            h = h_matrix[t, x]
            n = n_matrix[t, x]

            v_left = v_matrix[t, x - 1]
            v_center = v_matrix[t, x]
            v_right = v_matrix[t, x + 1]

            diffusion = alpha * (v_left - 2 * v_center + v_right)

            # hh currents
            i_Na = g_Na * (m**3) * h * (v - E_Na)
            i_K = g_K * (n**4) * (v - E_K)
            i_L = g_L * (v - E_L)

            i_ion = i_Na + i_K + i_L

            i_applied = 0.0
            if x == 1 and t < input_steps:
                i_applied = i_stim

            dv_dt = diffusion - (i_ion * dt) / c_m + (i_applied * dt) / c_m
            v_matrix[t + 1, x] = v + dv_dt

            # update gating variables
            am = alpha_m(v)
            bm = beta_m(v)
            ah = alpha_h(v)
            bh = beta_h(v)
            an = alpha_n(v)
            bn = beta_n(v)

            m_matrix[t + 1, x] = m + dt * (am * (1 - m) - bm * m)
            h_matrix[t + 1, x] = h + dt * (ah * (1 - h) - bh * h)
            n_matrix[t + 1, x] = n + dt * (an * (1 - n) - bn * n)

        # apply right boundary condition (sealed end)
        v_matrix[t + 1, n_x - 1] = v_matrix[t + 1, n_x - 2]
        h_matrix[t + 1, n_x - 1] = h_matrix[t + 1, n_x - 2]
        n_matrix[t + 1, n_x - 1] = n_matrix[t + 1, n_x - 2]
        m_matrix[t + 1, n_x - 1] = m_matrix[t + 1, n_x - 2]

        # apply left boundary condition (sealed end)
        v_matrix[t + 1, 0] = v_matrix[t + 1, 1]
        h_matrix[t + 1, 0] = h_matrix[t + 1, 1]
        n_matrix[t + 1, 0] = n_matrix[t + 1, 1]
        m_matrix[t + 1, 0] = m_matrix[t + 1, 1]

    fig, ax1 = plt.subplots(
        figsize=(10, 6),
    )

    ax1.imshow(
        v_matrix.T,
        aspect="auto",
        cmap="viridis",
        vmin=-80,
        vmax=40,
        origin="lower",
    )
    ax1.set_ylabel("Position")
    ax1.set_xlabel("Time")

    ax1.set_title("Action potential propagation")

    plt.show()


if __name__ == "__main__":
    main()
