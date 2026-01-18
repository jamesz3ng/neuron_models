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
        "refractory_period_ms": 5.0,  # absolute refractory period
        "spike_threshold_mv": -20.0,  # threshold for spike detection
    }


def _apply_refractory_period(
    v_in: np.ndarray,
    t_ms: np.ndarray,
    period_ms: float,
    threshold_mv: float,
    v_rest: float,
) -> tuple[np.ndarray, int]:
    """
    Filter input waveform to enforce absolute refractory period.

    Spikes arriving within `period_ms` of the last successful spike
    are suppressed (voltage set to v_rest while above threshold).

    Parameters
    ----------
    v_in : np.ndarray
        Input voltage waveform.
    t_ms : np.ndarray
        Time array in milliseconds.
    period_ms : float
        Absolute refractory period in ms.
    threshold_mv : float
        Voltage threshold for spike detection.
    v_rest : float
        Resting potential to use for suppressed spikes.

    Returns
    -------
    tuple[np.ndarray, int]
        (filtered_voltage, blocked_count)
    """
    v_out = v_in.copy()
    blocked_count = 0

    last_spike_time = -np.inf  # Allow first spike unconditionally
    in_spike = False
    spike_blocked = False

    for i in range(len(v_in)):
        above_threshold = v_in[i] >= threshold_mv

        if above_threshold and not in_spike:
            # Spike onset detected
            in_spike = True
            time_since_last = t_ms[i] - last_spike_time

            if time_since_last < period_ms:
                # Within refractory period - block this spike
                spike_blocked = True
                blocked_count += 1
            else:
                # Allow spike and reset timer
                spike_blocked = False
                last_spike_time = t_ms[i]

        if in_spike and spike_blocked and above_threshold:
            # Suppress voltage during blocked spike
            v_out[i] = v_rest

        if not above_threshold and in_spike:
            # Spike ended
            in_spike = False
            spike_blocked = False

    return v_out, blocked_count


def simulate_fast_model(
    *,
    v_input: np.ndarray,
    t_ms_input: np.ndarray,
    delay_ms: float,
    v_rest: float = -65.0,
    refractory_period_ms: float = 5.0,
    spike_threshold_mv: float = -20.0,
) -> dict:
    """
    Delay-line model with refractory period filtering.

    Filters input waveform to enforce abso thinlute refractory period,
    then shifts by delay_ms.

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
    refractory_period_ms : float
        Absolute refractory period in ms. Spikes arriving within this
        period after a successful spike are suppressed. Set to 0 to disable.
    spike_threshold_mv : float
        Voltage threshold for spike detection (mV).

    Returns
    -------
    dict with keys:
        - t_ms: time array (same as input)
        - V: delayed voltage waveform
        - delay_ms: the applied delay
        - delay_steps: delay in time steps
        - blocked_count: number of spikes suppressed by refractory filter
    """
    if len(v_input) != len(t_ms_input):
        raise ValueError("v_input and t_ms_input must have the same length")

    # Apply refractory period filtering
    if refractory_period_ms > 0:
        v_filtered, blocked_count = _apply_refractory_period(
            v_input, t_ms_input, refractory_period_ms, spike_threshold_mv, v_rest
        )
    else:
        v_filtered = v_input
        blocked_count = 0

    dt_ms = t_ms_input[1] - t_ms_input[0] if len(t_ms_input) > 1 else 1.0
    delay_steps = int(round(delay_ms / dt_ms))

    v_output = np.full_like(v_filtered, v_rest, dtype=float)

    if delay_steps < len(v_filtered) and delay_steps > 0:
        v_output[delay_steps:] = v_filtered[:-delay_steps]
    elif delay_steps == 0:
        v_output[:] = v_filtered[:]

    return {
        "t_ms": t_ms_input.copy(),
        "V": v_output,
        "delay_ms": float(delay_ms),
        "delay_steps": int(delay_steps),
        "blocked_count": blocked_count,
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
