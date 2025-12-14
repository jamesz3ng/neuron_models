import numpy as np

def alpha_n_safe(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10) + 1e-9)


def beta_n(V):
    return 0.125 * np.exp((V + 65) / -80)


def alpha_m_safe(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10) + 1e-9)


def beta_m(V):
    return 4.0 * np.exp((V + 65) / -18)


def alpha_h(V):
    return 0.07 * np.exp((V + 65) / -20)


def beta_h(V):
    return 1.0 / (1.0 + np.exp((V + 35) / -10))


def simulate_hh_cable(
    *,
    L: float = 5000.0,
    T: float = 40.0,
    dx: float = 10.0,
    dt: float = 0.001,
    lam: float = 200.0,
    tau: float = 1.0,
    c_m: float = 1.0,
    g_Na: float = 120.0,
    g_K: float = 36.0,
    g_L: float = 0.3,
    E_Na: float = 50.0,
    E_K: float = -77.0,
    E_L: float = -54.387,
    v_rest: float = -65.0,
    stim_duration: float = 5.0,
    stim_amplitude: float = 1000.0,
    stim_index: int = 1,
    store_history: bool = True,
    history_stride: int = 1,
):
    n_x = int(L / dx)
    n_t = int(T / dt)

    # diffusion coefficient
    alpha = (lam**2 * dt) / (tau * dx**2)

    if store_history:
        v_matrix = np.zeros((n_t, n_x))
        m_matrix = np.zeros((n_t, n_x))
        n_matrix = np.zeros((n_t, n_x))
        h_matrix = np.zeros((n_t, n_x))

        v_matrix[0, :] = v_rest
        m_matrix[0, :] = alpha_m_safe(v_rest) / (alpha_m_safe(v_rest) + beta_m(v_rest))
        h_matrix[0, :] = alpha_h(v_rest) / (alpha_h(v_rest) + beta_h(v_rest))
        n_matrix[0, :] = alpha_n_safe(v_rest) / (alpha_n_safe(v_rest) + beta_n(v_rest))

        input_steps = int(stim_duration / dt)
        for t in range(n_t - 1):
            v = v_matrix[t, :]
            m = m_matrix[t, :]
            h = h_matrix[t, :]
            n = n_matrix[t, :]

            v_inner = v[1:-1]
            v_left = v[:-2]
            v_right = v[2:]
            diffusion_term = alpha * (v_left - 2 * v_inner + v_right)

            i_Na = g_Na * (m**3) * h * (v - E_Na)
            i_K = g_K * (n**4) * (v - E_K)
            i_L = g_L * (v - E_L)
            i_ion = i_Na + i_K + i_L

            i_applied = np.zeros(n_x)
            if t < input_steps and 0 <= stim_index < n_x:
                i_applied[stim_index] = stim_amplitude

            v_matrix[t + 1, :] = v
            dv_dt_inner = (
                diffusion_term
                - (i_ion[1:-1] * dt) / c_m
                + (i_applied[1:-1] * dt) / c_m
            )
            v_matrix[t + 1, 1:-1] = v_inner + dv_dt_inner

            v_matrix[t + 1, 0] = v_matrix[t + 1, 1]
            v_matrix[t + 1, -1] = v_matrix[t + 1, -2]

            am = alpha_m_safe(v)
            bm = beta_m(v)
            ah = alpha_h(v)
            bh = beta_h(v)
            an = alpha_n_safe(v)
            bn = beta_n(v)

            m_matrix[t + 1, :] = m + dt * (am * (1 - m) - bm * m)
            h_matrix[t + 1, :] = h + dt * (ah * (1 - h) - bh * h)
            n_matrix[t + 1, :] = n + dt * (an * (1 - n) - bn * n)

        return {
            "alpha": float(alpha),
            "n_x": int(n_x),
            "n_t": int(n_t),
            "L": float(L),
            "T": float(T),
            "dx": float(dx),
            "dt": float(dt),
            "v_matrix": v_matrix[::history_stride, :],
        }

    v = np.full(n_x, v_rest, dtype=float)
    m = np.full(n_x, alpha_m_safe(v_rest) / (alpha_m_safe(v_rest) + beta_m(v_rest)))
    h = np.full(n_x, alpha_h(v_rest) / (alpha_h(v_rest) + beta_h(v_rest)))
    n = np.full(n_x, alpha_n_safe(v_rest) / (alpha_n_safe(v_rest) + beta_n(v_rest)))

    input_steps = int(stim_duration / dt)
    for t in range(n_t - 1):
        v_inner = v[1:-1]
        diffusion_term = alpha * (v[:-2] - 2 * v_inner + v[2:])

        i_Na = g_Na * (m**3) * h * (v - E_Na)
        i_K = g_K * (n**4) * (v - E_K)
        i_L = g_L * (v - E_L)
        i_ion = i_Na + i_K + i_L

        if t < input_steps and 0 <= stim_index < n_x:
            i_applied_inner = np.zeros(n_x - 2)
            if 1 <= stim_index < n_x - 1:
                i_applied_inner[stim_index - 1] = stim_amplitude
        else:
            i_applied_inner = 0.0

        dv_dt_inner = (
            diffusion_term - (i_ion[1:-1] * dt) / c_m + (i_applied_inner * dt) / c_m
        )
        v_next = v.copy()
        v_next[1:-1] = v_inner + dv_dt_inner
        v_next[0] = v_next[1]
        v_next[-1] = v_next[-2]

        am = alpha_m_safe(v)
        bm = beta_m(v)
        ah = alpha_h(v)
        bh = beta_h(v)
        an = alpha_n_safe(v)
        bn = beta_n(v)

        m = m + dt * (am * (1 - m) - bm * m)
        h = h + dt * (ah * (1 - h) - bh * h)
        n = n + dt * (an * (1 - n) - bn * n)
        v = v_next

    return {
        "alpha": float(alpha),
        "n_x": int(n_x),
        "n_t": int(n_t),
        "L": float(L),
        "T": float(T),
        "dx": float(dx),
        "dt": float(dt),
        "v_final": v,
    }


def main():
    import matplotlib.pyplot as plt

    result = simulate_hh_cable(store_history=True, history_stride=1)
    v_matrix = result["v_matrix"]
    alpha = result["alpha"]
    T = result["T"]
    L = result["L"]

    if alpha > 0.5:
        print(f"WARNING: alpha > 0.5 (alpha={alpha:.4f}).")
    else:
        print(f"Simulation alpha={alpha:.4f}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plot_step = max(1, int(v_matrix.shape[0] / 1000))

    img = ax1.imshow(
        v_matrix[::plot_step, :].T,
        aspect="auto",
        cmap="viridis",
        vmin=-80,
        vmax=40,
        origin="lower",
        extent=[0, T, 0, L],
    )

    plt.colorbar(img, label="Voltage (mV)")
    ax1.set_ylabel("Position (um)")
    ax1.set_xlabel("Time (ms)")
    ax1.set_title(f"AP Propagation (alpha={alpha:.2f})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
