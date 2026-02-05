import numpy as np

def _default_params():
    """
    Unit conventions for the public API:
    - Time is in seconds (`T_s`, `dt_s`, `stim_duration_s`).
    - Voltage is in millivolts (mV).
    - Length is in micrometers (um).

    Internally, some stability/scaling terms use millisecond time units; `dt_s` is
    converted to `dt_ms` for those calculations.
    """
    return {
        "L": 5000.0,  # um
        "T_s": 4e-2,  # 40 ms
        "dx": 10.0,
        "dt_s": 1e-6,  # 0.001 ms
        "lam": 200.0,
        "tau_ms": 1.0,
        "c_m": 1.0,
        "g_Na": 120.0,  # uS/cm^2
        "g_K": 36.0,
        "g_L": 0.3,
        "E_Na": 50.0,
        "E_K": -77.0,
        "E_L": -54.387,
        "v_rest": -65.0,
        "stim_duration_s": 5e-3,  # 5 ms
        "stim_amplitude": 1000.0,
        "stim_index": 1,
        "store_history": True,
        "history_stride": 1,
    }


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
    L: float | None = None,
    T_s: float | None = None,
    dx: float | None = None,
    dt_s: float | None = None,
    lam: float | None = None,
    tau_ms: float | None = None,
    c_m: float | None = None,
    g_Na: float | None = None,
    g_K: float | None = None,
    g_L: float | None = None,
    E_Na: float | None = None,
    E_K: float | None = None,
    E_L: float | None = None,
    v_rest: float | None = None,
    stim_duration_s: float | None = None,
    stim_amplitude: float | None = None,
    stim_index: int | None = None,
    stim_waveform: np.ndarray | None = None,
    t_s_stim: np.ndarray | None = None,
    store_history: bool | None = None,
    history_stride: int | None = None,
) -> dict:
    defaults = _default_params()
    L = defaults["L"] if L is None else L
    T_s = defaults["T_s"] if T_s is None else T_s
    dx = defaults["dx"] if dx is None else dx
    dt_s = defaults["dt_s"] if dt_s is None else dt_s
    lam = defaults["lam"] if lam is None else lam
    tau_ms = defaults["tau_ms"] if tau_ms is None else tau_ms
    c_m = defaults["c_m"] if c_m is None else c_m
    g_Na = defaults["g_Na"] if g_Na is None else g_Na
    g_K = defaults["g_K"] if g_K is None else g_K
    g_L = defaults["g_L"] if g_L is None else g_L
    E_Na = defaults["E_Na"] if E_Na is None else E_Na
    E_K = defaults["E_K"] if E_K is None else E_K
    E_L = defaults["E_L"] if E_L is None else E_L
    v_rest = defaults["v_rest"] if v_rest is None else v_rest
    stim_duration_s = (
        defaults["stim_duration_s"] if stim_duration_s is None else stim_duration_s
    )
    stim_amplitude = (
        defaults["stim_amplitude"] if stim_amplitude is None else stim_amplitude
    )
    stim_index = defaults["stim_index"] if stim_index is None else stim_index
    store_history = defaults["store_history"] if store_history is None else store_history
    history_stride = (
        defaults["history_stride"] if history_stride is None else history_stride
    )

    n_x = int(L / dx)
    n_t = int(T_s / dt_s)
    dt_ms = dt_s * 1e3

    # diffusion coefficient
    alpha = (lam**2 * dt_ms) / (tau_ms * dx**2)

    if store_history:
        v_matrix = np.zeros((n_t, n_x))
        m_matrix = np.zeros((n_t, n_x))
        n_matrix = np.zeros((n_t, n_x))
        h_matrix = np.zeros((n_t, n_x))

        v_matrix[0, :] = v_rest
        m_matrix[0, :] = alpha_m_safe(v_rest) / (alpha_m_safe(v_rest) + beta_m(v_rest))
        h_matrix[0, :] = alpha_h(v_rest) / (alpha_h(v_rest) + beta_h(v_rest))
        n_matrix[0, :] = alpha_n_safe(v_rest) / (alpha_n_safe(v_rest) + beta_n(v_rest))

        # Prepare stimulus boundary
        if stim_waveform is not None and t_s_stim is not None:
            t_sim = np.arange(n_t) * dt_s
            i_applied_boundary = np.interp(t_sim, t_s_stim, stim_waveform, left=0.0, right=0.0)
            use_waveform_stim = True
        else:
            use_waveform_stim = False
            input_steps = int(stim_duration_s / dt_s)

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
            if use_waveform_stim:
                if 0 <= stim_index < n_x:
                    i_applied[stim_index] = i_applied_boundary[t]
            elif t < input_steps and 0 <= stim_index < n_x:
                i_applied[stim_index] = stim_amplitude

            v_matrix[t + 1, :] = v
            dv_dt_inner = (
                diffusion_term
                - (i_ion[1:-1] * dt_ms) / c_m
                + (i_applied[1:-1] * dt_ms) / c_m
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

            m_matrix[t + 1, :] = m + dt_ms * (am * (1 - m) - bm * m)
            h_matrix[t + 1, :] = h + dt_ms * (ah * (1 - h) - bh * h)
            n_matrix[t + 1, :] = n + dt_ms * (an * (1 - n) - bn * n)

        return {
            "alpha": float(alpha),
            "n_x": int(n_x),
            "n_t": int(n_t),
            "L": float(L),
            "T_s": float(T_s),
            "dx": float(dx),
            "dt_s": float(dt_s),
            "v_matrix": v_matrix[::history_stride, :],
        }

    v = np.full(n_x, v_rest, dtype=float)
    m = np.full(n_x, alpha_m_safe(v_rest) / (alpha_m_safe(v_rest) + beta_m(v_rest)))
    h = np.full(n_x, alpha_h(v_rest) / (alpha_h(v_rest) + beta_h(v_rest)))
    n = np.full(n_x, alpha_n_safe(v_rest) / (alpha_n_safe(v_rest) + beta_n(v_rest)))

    # Prepare stimulus boundary
    if stim_waveform is not None and t_s_stim is not None:
        t_sim = np.arange(n_t) * dt_s
        i_applied_boundary = np.interp(t_sim, t_s_stim, stim_waveform, left=0.0, right=0.0)
        use_waveform_stim = True
    else:
        use_waveform_stim = False
        input_steps = int(stim_duration_s / dt_s)

    for t in range(n_t - 1):
        v_inner = v[1:-1]
        diffusion_term = alpha * (v[:-2] - 2 * v_inner + v[2:])

        i_Na = g_Na * (m**3) * h * (v - E_Na)
        i_K = g_K * (n**4) * (v - E_K)
        i_L = g_L * (v - E_L)
        i_ion = i_Na + i_K + i_L

        i_applied = np.zeros(n_x)
        if use_waveform_stim:
            if 0 <= stim_index < n_x:
                i_applied[stim_index] = i_applied_boundary[t]
        elif t < input_steps and 0 <= stim_index < n_x:
            i_applied[stim_index] = stim_amplitude

        i_applied_inner = i_applied[1:-1]

        dv_dt_inner = (
            diffusion_term
            - (i_ion[1:-1] * dt_ms) / c_m
            + (i_applied_inner * dt_ms) / c_m
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

        m = m + dt_ms * (am * (1 - m) - bm * m)
        h = h + dt_ms * (ah * (1 - h) - bh * h)
        n = n + dt_ms * (an * (1 - n) - bn * n)
        v = v_next

    return {
        "alpha": float(alpha),
        "n_x": int(n_x),
        "n_t": int(n_t),
        "L": float(L),
        "T_s": float(T_s),
        "dx": float(dx),
        "dt_s": float(dt_s),
        "v_final": v,
    }


def main():
    import matplotlib.pyplot as plt

    result = simulate_hh_cable(store_history=True, history_stride=1)
    v_matrix = result["v_matrix"]
    alpha = result["alpha"]
    T_s = result["T_s"]
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
        extent=[0, T_s, 0, L],
    )

    plt.colorbar(img, label="Voltage (mV)")
    ax1.set_ylabel("Position (um)")
    ax1.set_xlabel("Time (s)")
    ax1.set_title(f"AP Propagation (alpha={alpha:.2f})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
