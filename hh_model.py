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

# rate functions
# k activation


def alpha_n(V):
    return -0.01 * (V + 55) / (np.exp((V + 55) / -10) - 1)


def beta_n(V):
    return 0.125 * np.exp((V + 65) / 80)


def alpha_m(V):
    return -0.1 * (V + 40) / (np.exp((V + 40) / -10) - 1)


def beta_m(V):
    return 4 * np.exp((V + 65) / -18)


def alpha_h(V):
    return 0.07 * np.exp((V + 65) / -20)


def beta_h(V):
    return 1 / (1 + np.exp((V + 35) / -10))


def main():
    dt = 0.01  # ms
    T = 100.0
    t = np.arange(0, T, dt)

    V = np.zeros(len(t))
    m = np.zeros(len(t))
    h = np.zeros(len(t))
    n = np.zeros(len(t))

    # set initial resting positions
    V[0] = -65.0
    m[0] = 0.0529
    h[0] = 0.5961
    n[0] = 0.3177

    I_inj = np.zeros(len(t))
    I_inj[int(20 / dt) : int(21 / dt)] = 200

    # Using the Euler forward method
    for i in range(1, len(t)):
        V_old = V[i - 1]
        m_old = m[i - 1]
        h_old = h[i - 1]
        n_old = n[i - 1]

        I_Na = g_Na * (m_old**3) * h_old * (V_old - E_Na)
        I_K = g_K * (n_old**4) * (V_old - E_K)
        I_L = g_L * (V_old - E_L)

        dm = alpha_m(V_old) * (1 - m_old) - beta_m(V_old) * m_old
        dh = alpha_h(V_old) * (1 - h_old) - beta_h(V_old) * h_old
        dn = alpha_n(V_old) * (1 - n_old) - beta_n(V_old) * n_old

        m[i] = m_old + dm * dt
        h[i] = h_old + dh * dt
        n[i] = n_old + dn * dt

        dV = (I_inj[i - 1] - I_Na - I_K - I_L) / C_m
        V[i] = V_old + dV * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    ax1.plot(t, V, color="blue")
    # ax1.plot(t, m_p, color="green", alpha=0.3, label="m gate")
    # ax1.plot(t, n_p, color="red", alpha=0.3, label="n gate")
    # ax1.plot(t, h_p, color="purple", alpha=0.3, label="h gate")

    ax1_right = ax1.twinx()
    ax1_right.set_ylabel("Gating Probability")
    ax1_right.set_ylim(0, 1.1)

    gates = {"m": (m, "green"), "n": (n, "red"), "h": (h, "purple")}

    for name, (data, color) in gates.items():
        ax1_right.plot(t, data, color=color, alpha=0.3, label=f"{name} gate")

    ax1_right.legend()

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Membrane Potential (mV)")
    ax1.set_title("Action Potential Propagation")
    ax1.grid(True)

    ax2.plot(t, I_inj, color="Red")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Injeted Current")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
