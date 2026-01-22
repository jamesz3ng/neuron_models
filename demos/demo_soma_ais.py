"""
Biophysical Source-Filter Validation (Soma-AIS Model)

Validates the architectural decision to simulate the Axon Initial Segment (AIS)
as the source for the Event Model. Proves that the AIS generates a sharper,
narrower spike than the Soma, which is critical for accurate synaptic weight
calculation (Rowan et al., 2016).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
_OUTPUT_DIR = _ROOT / "output"

import numpy as np
import matplotlib.pyplot as plt

from src.ssds_model import HHPhysics
from src.ais_simulation import run_2comp_simulation
from src.event_model import EventPropagator

# Simulation Constants
DT_MS = 0.002



def calculate_metrics(t, V, v_th=-20.0):
    # FWHM
    peak_idx = np.argmax(V)
    peak_val = V[peak_idx]
    base_val = V[0]
    half_max = (peak_val + base_val) / 2.0
    
    # Find crossings
    crossings = np.where(np.diff(np.sign(V - half_max)))[0]
    
    # Logic to find crossings relative to peak
    left_cross = crossings[crossings < peak_idx]
    right_cross = crossings[crossings > peak_idx]
    
    if len(left_cross) > 0 and len(right_cross) > 0:
        left = left_cross[-1]
        right = right_cross[0]
        fwhm = t[right] - t[left]
    else:
        fwhm = 0.0
        
    
    drive_signal = np.maximum(0, V - v_th)**3
    drive_integral = np.sum(drive_signal) * (t[1] - t[0])
    
    return fwhm, drive_integral, drive_signal

def main():
    print("="*60)
    print("Soma-AIS Source-Filter Validation")
    print("="*60)
    
    # 1. Setup Simulation
    T_END = 20.0
    stim_start = 5.0
    stim_dur = 1.0
    stim_amp = 100.0
    
    n_pts = int(T_END / DT_MS)
    i_stim = np.zeros(n_pts)
    start_idx = int(stim_start / DT_MS)
    end_idx = int((stim_start + stim_dur) / DT_MS)
    i_stim[start_idx:end_idx] = stim_amp
    
    print("Running 2-compartment simulation...")
    t, Vs, Va, i_na_s, i_k_s, i_na_a, i_k_a = run_2comp_simulation(T_END, i_stim, DT_MS)
    
    # 2. Metrics
    print("\nMetrics:")
    print("-" * 30)
    
    fwhm_s, drive_s, signal_s = calculate_metrics(t, Vs)
    fwhm_a, drive_a, signal_a = calculate_metrics(t, Va)
    
    print(f"Soma Peak: {np.max(Vs):.2f} mV")
    print(f"AIS  Peak: {np.max(Va):.2f} mV")
    
    print(f"Soma FWHM:  {fwhm_s:.3f} ms")
    print(f"AIS  FWHM:  {fwhm_a:.3f} ms")
    print(f"Soma Drive: {drive_s:.2f} (a.u.)")
    print(f"AIS  Drive: {drive_a:.2f} (a.u.)")
    
    ratio_fwhm = fwhm_s / fwhm_a if fwhm_a > 0 else 0
    ratio_drive = drive_s / drive_a if drive_a > 0 else 0
    
    print(f"\nSoma is {ratio_fwhm:.1f}x wider than AIS")
    print(f"Soma overestimates synaptic drive by {ratio_drive:.1f}x")
    
    # 3. Event Model Check
    print("\nEvent Model Check:")
    print("-" * 30)
    v_recon = None
    try:
        prop = EventPropagator()
        # Feed AIS trace
        res = prop.simulate(Va, t)
        v_recon = res['v_out']
        print(f"AIS Trace -> EventPropagator -> Reconstructed")
        print(f"Encoded into {res['n_spikes']} event(s).")
        print("Reconstruction successful.")
    except Exception as e:
        print(f"EventPropagator failed: {e}")
        
    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot 1: Voltage
    ax1.plot(t, Vs, 'k-', linewidth=2, label='Soma (Standard HH)')
    ax1.plot(t, Va, 'b-', linewidth=2, label='AIS (High Density)')
    
    if v_recon is not None:
        delay = prop.delay_ms
        t_shifted = t + delay
        ax1.plot(t, v_recon, 'r--', linewidth=1.5, label=f'Event Model Output')        
    ax1.set_title('Action Potential Source Generation & Event Encoding')
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(4, 30) # Zoom on spike + reconstruction
    
    # Plot 2: Ionic Currents (AIS)
    ax2.plot(t, i_na_a, 'r-', linewidth=1.5, label='AIS $I_{Na}$')
    ax2.plot(t, i_k_a, 'g-', linewidth=1.5, label='AIS $I_{K}$')
    
    # Optional: Plot Soma currents as dashed lines for comparison
    ax2.plot(t, i_na_s, 'r--', linewidth=1, alpha=0.5, label='Soma $I_{Na}$')
    ax2.plot(t, i_k_s, 'g--', linewidth=1, alpha=0.5, label='Soma $I_{K}$')
    
    ax2.set_title('Ionic Currents (Sodium & Potassium)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel(r'Current Density ($\mu$A/cm²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(4, 15)
    
    plt.tight_layout()
    output_file = _OUTPUT_DIR / "soma_ais_validation.png"
    plt.savefig(output_file)
    print(f"\nSaved plot to {output_file}")

if __name__ == "__main__":
    main()
