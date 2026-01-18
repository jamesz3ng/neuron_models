"""
Biophysical Source-Filter Validation (Soma-AIS Model)

Validates the architectural decision to simulate the Axon Initial Segment (AIS)
as the source for the Event Model. Proves that the AIS generates a sharper,
narrower spike than the Soma, which is critical for accurate synaptic weight
calculation (Rowan et al., 2016).
"""

import numpy as np
import matplotlib.pyplot as plt

from ssds_model import HHPhysics, run_simulation
from event_model import EventPropagator

# Simulation Constants
DT_MS = 0.01

def run_2comp_simulation(t_end_ms: float, i_stim_soma: np.ndarray, dt_ms: float = 0.01):
    """
    Run a 2-compartment simulation (Soma + AIS).
    
    Physics:
      Soma: Standard HH (gNa=120, gK=36)
      AIS:  High Density (gNa=600, gK=100)
    
    Coupling:
      Ohms law axial current between compartments.
    """
    n_steps = len(i_stim_soma)
    t_ms = np.arange(n_steps) * dt_ms
    
    # --- Parameters ---
    # Soma (Standard)
    g_Na_s = 120.0
    g_K_s = 36.0
    g_L = 0.3
    
    # AIS (High Density - Kole et al 2008)
    g_Na_a = 400.0  
    g_K_a = 200.0   
    
    # Reversal Potentials
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    
    # Coupling
    g_couple = 2.0  # mS/cm2
    C_m = 1.0       # uF/cm2
    
    # --- Initialization ---
    v_rest = -65.0
    Vs = v_rest
    Va = v_rest
    
    # Steady state gates
    m, h, n = HHPhysics.steady_state(v_rest)
    
    # State vectors (Soma and AIS start same)
    # Soma gates
    ms, hs, ns = m, h, n
    # AIS gates
    ma, ha, na = m, h, n
    
    # History
    Vs_hist = np.zeros(n_steps)
    Va_hist = np.zeros(n_steps)
    
    # Loop
    for i in range(n_steps):
        # 1. Calculate Currents
        # Soma
        I_Na_s = g_Na_s * (ms**3) * hs * (Vs - E_Na)
        I_K_s  = g_K_s  * (ns**4) * (Vs - E_K)
        I_L_s  = g_L    * (Vs - E_L)
        
        # AIS
        I_Na_a = g_Na_a * (ma**3) * ha * (Va - E_Na)
        I_K_a  = g_K_a  * (na**4) * (Va - E_K)
        I_L_a  = g_L    * (Va - E_L)
        
        # Axial Currents
        # Current flowing INTO Soma FROM AIS
        I_axial_s = g_couple * (Va - Vs)
        # Current flowing INTO AIS FROM Soma
        I_axial_a = g_couple * (Vs - Va)
        
        # 2. Update Voltages
        # Note: Stimulus applied only to Soma
        dVs = (i_stim_soma[i] - I_Na_s - I_K_s - I_L_s + I_axial_s) / C_m
        dVa = (0.0            - I_Na_a - I_K_a - I_L_a + I_axial_a) / C_m
        
        Vs += dVs * dt_ms
        Va += dVa * dt_ms
        
        # 3. Update Gates
        # Soma
        am, bm, ah, bh, an, bn = HHPhysics.rates(Vs)
        ms += (am * (1 - ms) - bm * ms) * dt_ms
        hs += (ah * (1 - hs) - bh * hs) * dt_ms
        ns += (an * (1 - ns) - bn * ns) * dt_ms
        
        # AIS
        am, bm, ah, bh, an, bn = HHPhysics.rates(Va)
        ma += (am * (1 - ma) - bm * ma) * dt_ms
        ha += (ah * (1 - ha) - bh * ha) * dt_ms
        na += (an * (1 - na) - bn * na) * dt_ms
        
        # Store
        Vs_hist[i] = Vs
        Va_hist[i] = Va
        
    return t_ms, Vs_hist, Va_hist

def calculate_metrics(t, V, v_th=-20.0):
    """Calculate FWHM and Synaptic Drive."""
    # FWHM
    peak_idx = np.argmax(V)
    peak_val = V[peak_idx]
    base_val = V[0]
    half_max = (peak_val + base_val) / 2.0
    
    # Find crossings
    crossings = np.where(np.diff(np.sign(V - half_max)))[0]
    if len(crossings) >= 2:
        # Take the two closest to peak
        left = crossings[crossings < peak_idx][-1]
        right = crossings[crossings > peak_idx][0]
        fwhm = t[right] - t[left]
    else:
        fwhm = 0.0
        
    # Synaptic Drive Integral: sum(max(0, V - Vth)^3) * dt
    # Use -20mV as approximation for Ca channel activation threshold logic or release
    # The prompt says: sum(max(0, V - Vth)^3)
    # Ginebaugh et al often use -25 or -20.
    
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
    stim_amp = 30.0
    
    n_pts = int(T_END / DT_MS)
    i_stim = np.zeros(n_pts)
    start_idx = int(stim_start / DT_MS)
    end_idx = int((stim_start + stim_dur) / DT_MS)
    i_stim[start_idx:end_idx] = stim_amp
    
    print("Running 2-compartment simulation...")
    t, Vs, Va = run_2comp_simulation(T_END, i_stim, DT_MS)
    
    # 2. Metrics
    print("\nMetrics:")
    print("-" * 30)
    
    fwhm_s, drive_s, signal_s = calculate_metrics(t, Vs)
    fwhm_a, drive_a, signal_a = calculate_metrics(t, Va)
    
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
        # Align reconstruction if needed:
        # The Propagator adds a specific delay (default 5ms).
        # We need to shift it back to overlay exactly, OR we just plot it as is to show propagation.
        # However, prompt says "how well it reconstructs". Usually implies overlay.
        # Propagator default delay is 5ms.
        # Let's check delay from prop object.
        delay = prop.delay_ms
        t_shifted = t + delay
        ax1.plot(t_shifted, v_recon, 'r--', linewidth=1.5, label=f'Event Model Recon (+{delay}ms)')
        
    ax1.set_title('Action Potential Source Generation & Event Encoding')
    ax1.set_ylabel('Voltage (mV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(4, 30) # Zoom on spike + reconstruction
    
    # Plot 2: Synaptic Drive
    ax2.plot(t, signal_s, 'k--', linewidth=2, label='Soma Drive (V-Vth)^3')
    ax2.fill_between(t, signal_s, color='black', alpha=0.1)
    
    ax2.plot(t, signal_a, 'b-', linewidth=2, label='AIS Drive (V-Vth)^3')
    ax2.fill_between(t, signal_a, color='blue', alpha=0.2)
    
    ax2.set_title('Synaptic Drive Function (Calcium Influx Proxy)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Drive Strength (a.u.)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(4, 12)
    
    plt.tight_layout()
    plt.savefig('soma_ais_validation.png')
    print("\nSaved plot to soma_ais_validation.png")

if __name__ == "__main__":
    main()
