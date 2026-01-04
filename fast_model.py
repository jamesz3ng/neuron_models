import numpy as np


def _default_params():
    """
    Unit conventions for the public API:
    - Time is in milliseconds (ms).
    - Length is in micrometers (μm).
    - Conduction velocity is in μm/ms.
    """
    return {
        "c": 100.0,  # conduction velocity μm/ms (typical unmyelinated axon)
    }


def simulate_fast_model(
    *,
    v_input: np.ndarray,
    t_ms_input: np.ndarray,
    delay_ms: float,
    v_rest: float = -65.0,
) -> dict:
    """
    Pure delay-line model. Shifts input waveform by delay_ms.

    Parameters
    ----------
    v_input : np.ndarray
        Input voltage waveform (e.g., from HH model at stimulus site).
    t_ms_input : np.ndarray
        Time array in milliseconds corresponding to v_input.
    delay_ms : float
        Delay to apply in milliseconds (typically L / c).
    v_rest : float
        Resting potential to use for time points before signal arrives.

    Returns
    -------
    dict with keys:
        - t_ms: time array (same as input)
        - V: delayed voltage waveform
        - delay_ms: the applied delay
    """
    if len(v_input) != len(t_ms_input):
        raise ValueError("v_input and t_ms_input must have the same length")

    dt_ms = t_ms_input[1] - t_ms_input[0] if len(t_ms_input) > 1 else 1.0
    delay_steps = int(round(delay_ms / dt_ms))

    v_output = np.full_like(v_input, v_rest, dtype=float)

    if delay_steps < len(v_input) and delay_steps > 0:
        v_output[delay_steps:] = v_input[:-delay_steps]
    elif delay_steps == 0:
        v_output[:] = v_input[:]

    return {
        "t_ms": t_ms_input.copy(),
        "V": v_output,
        "delay_ms": float(delay_ms),
        "delay_steps": int(delay_steps),
    }


def _gaussian_input(
    t_ms: np.ndarray, peak_time_ms: float, width_ms: float, amplitude: float
):
    """Generate a Gaussian input pulse for demo purposes."""
    val = amplitude * np.exp(-((t_ms - peak_time_ms) ** 2) / (2 * width_ms**2))
    return np.where(val < 1e-5, 0, val)


def main():
    import matplotlib.pyplot as plt

    # Demo configuration
    L_um = 10000.0  # 10 mm in μm
    defaults = _default_params()
    c = defaults["c"]  # μm/ms
    T_ms = 500.0
    dt_ms = 1.0

    delay_ms = L_um / c
    t_ms = np.arange(0, T_ms, dt_ms)

    # Generate demo Gaussian input
    v_source = _gaussian_input(t_ms, peak_time_ms=50.0, width_ms=10.0, amplitude=100.0)

    # Run simulation
    result = simulate_fast_model(
        v_input=v_source,
        t_ms_input=t_ms,
        delay_ms=delay_ms,
    )

    v_output = result["V"]
    print(f"Delay is {delay_ms:.2f} ms ({result['delay_steps']} time steps)")

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(t_ms, v_source, "b--", label="Input (Source)")
    plt.plot(t_ms, v_output, "r-", lw=2, label="Output (Delayed)")

    plt.title(f"Exact Delay Model (Delay = {delay_ms:.1f} ms)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
