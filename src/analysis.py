
import numpy as np

SPIKE_THRESHOLD_MV = -20.0

def find_spike_peaks(
    V: np.ndarray, 
    threshold_mv: float = SPIKE_THRESHOLD_MV
) -> list[int]:
    """Find indices of spike peaks in voltage trace."""
    peaks = []
    in_spike = False
    max_v = -np.inf
    max_idx = 0

    for i in range(len(V)):
        if V[i] >= threshold_mv and not in_spike:
            in_spike = True
            max_v = V[i]
            max_idx = i
        elif in_spike and V[i] >= threshold_mv:
            if V[i] > max_v:
                max_v = V[i]
                max_idx = i
        elif in_spike and V[i] < threshold_mv:
            peaks.append(max_idx)
            in_spike = False
            max_v = -np.inf

    return peaks

def extract_aligned_spike(
    V: np.ndarray, 
    peak_idx: int, 
    pre_points: int, 
    post_points: int
) -> np.ndarray | None:
    """Extract spike waveform aligned to peak. Returns None if out of bounds."""
    start_idx = peak_idx - pre_points
    end_idx = peak_idx + post_points

    if start_idx < 0 or end_idx > len(V):
        return None

    return V[start_idx:end_idx].copy()

def measure_spike_width(
    V: np.ndarray, 
    peak_idx: int, 
    dt_ms: float
) -> float:
    """
    Measure spike width at half-maximum (FWHM).

    Returns width in ms, or -1 if measurement fails.
    """
    peak_v = V[peak_idx]
    baseline = -65.0  # Approximate resting potential
    half_max = (peak_v + baseline) / 2.0

    # Find left crossing
    left_idx = peak_idx
    while left_idx > 0 and V[left_idx] > half_max:
        left_idx -= 1

    # Find right crossing
    right_idx = peak_idx
    while right_idx < len(V) - 1 and V[right_idx] > half_max:
        right_idx += 1

    if left_idx == 0 or right_idx == len(V) - 1:
        return -1.0

    width_ms = (right_idx - left_idx) * dt_ms
    return width_ms
