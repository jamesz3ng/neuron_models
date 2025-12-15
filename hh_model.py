import numpy as np

# https://goldmanlab.faculty.ucdavis.edu/wp-content/uploads/sites/263/2016/07/HodgkinHuxley.pdf
# values taken from above converted to SI units

def _default_params():
    """
    Unit conventions for the public API:
    - Time is in seconds (`T_s`, `dt_s`, `stim_*_s`).
    - Voltage is in millivolts (mV).
    - Length is in centimeters (cm).

    Internally, the classic Hodgkin-Huxley rate equations are advanced in milliseconds by
    converting `dt_s` -> `dt_ms` (and similarly for stimulus times).
    """
    return {
        "C_m": 1.0,  # nF/mm^2
        "g_Na": 120.0,  # uS/cm^2
        "g_K": 36.0,
        "g_L": 0.3,
        "E_Na": 50.0,  # mV
        "E_K": -77.0,
        "E_L": -54.387,
        "length": 2.0,  # cm
        "diameter": 0.05,
        "r_a": 35.4,  # axial resistivity
        "n_spatial": 100,
        "dt_s": 1e-5,  # 0.01 ms
        "T_s": 5e-2,  # 50 ms
        "stim_start_s": 2e-2,  # 20 ms
        "stim_end_s": 3e-2,  # 30 ms
    }

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


def simulate_hh_model(
    *,
    C_m: float | None = None,
    g_Na: float | None = None,
    g_K: float | None = None,
    g_L: float | None = None,
    E_Na: float | None = None,
    E_K: float | None = None,
    E_L: float | None = None,
    length: float | None = None,
    diameter: float | None = None,
    r_a: float | None = None,
    n_spatial: int | None = None,
    dt_s: float | None = None,
    T_s: float | None = None,
    v_rest: float = -65.0,
    stim_start_s: float | None = None,
    stim_end_s: float | None = None,
    stim_amplitude: float = 20.0,
    stim_index: int = 0,
    store_history: bool = True,
    history_stride: int = 1,
):
    defaults = _default_params()
    C_m = defaults["C_m"] if C_m is None else C_m
    g_Na = defaults["g_Na"] if g_Na is None else g_Na
    g_K = defaults["g_K"] if g_K is None else g_K
    g_L = defaults["g_L"] if g_L is None else g_L
    E_Na = defaults["E_Na"] if E_Na is None else E_Na
    E_K = defaults["E_K"] if E_K is None else E_K
    E_L = defaults["E_L"] if E_L is None else E_L
    length = defaults["length"] if length is None else length
    diameter = defaults["diameter"] if diameter is None else diameter
    r_a = defaults["r_a"] if r_a is None else r_a
    n_spatial = defaults["n_spatial"] if n_spatial is None else n_spatial
    dt_s = defaults["dt_s"] if dt_s is None else dt_s
    T_s = defaults["T_s"] if T_s is None else T_s
    stim_start_s = defaults["stim_start_s"] if stim_start_s is None else stim_start_s
    stim_end_s = defaults["stim_end_s"] if stim_end_s is None else stim_end_s

    dt_ms = dt_s * 1e3
    T_ms = T_s * 1e3

    radius = diameter / 2
    dx = length / n_spatial
    diffusion_coeff = radius / (2 * r_a * dx**2 + C_m)

    n_time = int(T_s / dt_s)

    stim_start_idx = int(stim_start_s / dt_s)
    stim_end_idx = int(stim_end_s / dt_s)

    if store_history:
        t_s = np.arange(n_time) * dt_s
        x = np.linspace(0, length, n_spatial)

        V = np.zeros((n_time, n_spatial))
        m = np.zeros((n_time, n_spatial))
        h = np.zeros((n_time, n_spatial))
        n_gate = np.zeros((n_time, n_spatial))

        V[0, :] = v_rest
        m[0, :] = 0.0529
        h[0, :] = 0.5961
        n_gate[0, :] = 0.3177

        for i in range(1, n_time):
            V_old = V[i - 1, :]
            m_old = m[i - 1, :]
            h_old = h[i - 1, :]
            n_old = n_gate[i - 1, :]

            I_Na = g_Na * (m_old**3) * h_old * (V_old - E_Na)
            I_K = g_K * (n_old**4) * (V_old - E_K)
            I_L = g_L * (V_old - E_L)

            dm = alpha_m(V_old) * (1 - m_old) - beta_m(V_old) * m_old
            dh = alpha_h(V_old) * (1 - h_old) - beta_h(V_old) * h_old
            dn = alpha_n(V_old) * (1 - n_old) - beta_n(V_old) * n_old

            m[i, :] = m_old + dm * dt_ms
            h[i, :] = h_old + dh * dt_ms
            n_gate[i, :] = n_old + dn * dt_ms

            d2V = np.zeros(n_spatial)
            d2V[1:-1] = V_old[2:] - 2 * V_old[1:-1] + V_old[:-2]
            d2V[0] = V_old[1] - V_old[0]
            d2V[-1] = V_old[-2] - V_old[-1]

            I_inj = np.zeros(n_spatial)
            if stim_start_idx <= i - 1 < stim_end_idx and 0 <= stim_index < n_spatial:
                I_inj[stim_index] = stim_amplitude

            dV = (diffusion_coeff * d2V + I_inj - I_Na - I_K - I_L) / C_m
            V[i, :] = V_old + dV * dt_ms

        return {
            "t_s": t_s[::history_stride],
            "x": x,
            "V": V[::history_stride, :],
            "length": float(length),
            "dt_s": float(dt_s),
            "T_s": float(T_s),
            "n_spatial": int(n_spatial),
            "n_t": int(n_time),
        }

    V = np.full(n_spatial, v_rest, dtype=float)
    m = np.full(n_spatial, 0.0529, dtype=float)
    h = np.full(n_spatial, 0.5961, dtype=float)
    n_gate = np.full(n_spatial, 0.3177, dtype=float)

    for i in range(1, n_time):
        I_Na = g_Na * (m**3) * h * (V - E_Na)
        I_K = g_K * (n_gate**4) * (V - E_K)
        I_L = g_L * (V - E_L)

        dm = alpha_m(V) * (1 - m) - beta_m(V) * m
        dh = alpha_h(V) * (1 - h) - beta_h(V) * h
        dn = alpha_n(V) * (1 - n_gate) - beta_n(V) * n_gate

        m = m + dm * dt_ms
        h = h + dh * dt_ms
        n_gate = n_gate + dn * dt_ms

        d2V = np.zeros(n_spatial)
        d2V[1:-1] = V[2:] - 2 * V[1:-1] + V[:-2]
        d2V[0] = V[1] - V[0]
        d2V[-1] = V[-2] - V[-1]

        I_inj = np.zeros(n_spatial)
        if stim_start_idx <= i - 1 < stim_end_idx and 0 <= stim_index < n_spatial:
            I_inj[stim_index] = stim_amplitude

        dV = (diffusion_coeff * d2V + I_inj - I_Na - I_K - I_L) / C_m
        V = V + dV * dt_ms

    return {
        "V_final": V,
        "length": float(length),
        "dt_s": float(dt_s),
        "T_s": float(T_s),
        "n_spatial": int(n_spatial),
        "n_t": int(n_time),
    }


def main():
    import matplotlib.pyplot as plt

    result = simulate_hh_model(store_history=True, history_stride=1)
    t_s = result["t_s"]
    x = result["x"]
    V = result["V"]
    length = result["length"]
    dt_s = result["dt_s"]

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

    ax2 = plt.subplot(3, 1, 2)
    n_spatial = V.shape[1]
    positions_to_plot = [
        0,
        n_spatial // 4,
        n_spatial // 2,
        3 * n_spatial // 4,
        n_spatial - 1,
    ]
    for pos in positions_to_plot:
        ax2.plot(t_s * 1e3, V[:, pos], label=f"x = {x[pos]:.2f} cm")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Membrane Potential (mV)")
    ax2.set_title("Voltage at Different Positions")
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(3, 1, 3)
    times_to_plot = [10, 20, 25, 30, 35]
    for time_ms in times_to_plot:
        idx = int((time_ms * 1e-3) / dt_s)
        if idx < len(t_s):
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
