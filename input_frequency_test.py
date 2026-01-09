import numpy as np

import ssds_model
from ssds_model import _simulate_and_harvest

# Fix Amplitude to the one that worked best
ssds_model.STIM_AMPLITUDE = 30.0


def run_freq_test(freq_val):
    print(f"\n--- Testing FREQUENCY = {freq_val} Hz ---")

    # Setup pulse train
    isi = 1000.0 / freq_val
    times = [5.0 + i * isi for i in range(10)]

    # Run
    try:
        spikes = _simulate_and_harvest(times)
    except TypeError:
        # Handle different function signatures if necessary
        spikes = _simulate_and_harvest(times, t_end_ms=times[-1] + 20.0)

    if not spikes:
        print("  No spikes detected!")
        return

    n_found = len(spikes)
    print(f"  Spikes detected: {n_found}/10")

    if n_found < 2:
        return

    # Analyze First vs Last DETECTED spike
    first_peak = np.max(spikes[0]["waveform"])
    last_peak = np.max(spikes[-1]["waveform"])

    print(f"  Spike 1 Peak:  {first_peak:.2f} mV")
    print(f"  Last Spike Pk: {last_peak:.2f} mV")
    drop = first_peak - last_peak
    pct = (drop / first_peak) * 100
    print(f"  Drop:          {drop:.2f} mV ({pct:.1f}%)")


if __name__ == "__main__":
    # Fine-grained sweep near the limit
    for f in [
        50,
        60,
        65,
        70,
        80,
        85,
        88,
        90,
        92,
        95,
        100,
        103,
        105,
        107,
        110,
        113,
        116,
        120,
        123,
        126,
        130,
        133,
        136,
        140,
        144,
        147,
        150,
    ]:
        run_freq_test(f)
