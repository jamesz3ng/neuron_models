
import numpy as np
from .physics import HHPhysics

def run_2comp_simulation(
    t_end_ms: float, 
    i_stim_soma: np.ndarray, 
    dt_ms: float = 0.002,
    # Scaling Parameters
    g_na_s_scale: float = 1.0, 
    g_k_s_scale: float = 1.0,
    g_na_a_scale: float = 1.0, 
    g_k_a_scale: float = 1.0,
    g_couple_scale: float = 1.0,
    # Kinetic Scaling
    tau_m_scale_ais: float = 0.8, 
    tau_h_scale_ais: float = 1.0, 
    tau_n_scale_ais: float = 0.5
):
    """
    Run a 2-compartment simulation (Soma + AIS).
    
    Physics:
      Soma: Standard HH (gNa=120, gK=36) * scale
      AIS:  High Density (gNa=1000, gK=200) * scale
    
    Coupling:
      Ohms law axial current between compartments.
    """
    n_steps = len(i_stim_soma)
    t_ms = np.arange(n_steps) * dt_ms
    
    # --- Parameters ---
    # Soma (Standard)
    g_Na_s = 120.0 * g_na_s_scale
    g_K_s = 36.0 * g_k_s_scale
    g_L = 0.3
    
    # AIS (High Density - Tuned for Sharpening)
    g_Na_a = 1300.0 * g_na_a_scale
    g_K_a = 300.0 * g_k_a_scale
    
    # Reversal Potentials
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    
    # Coupling
    g_couple = 2.0 * g_couple_scale # mS/cm2
    C_m = 1.0       # uF/cm2
    inv_C_m = 1.0 / C_m
    
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
    i_na_s_hist = np.zeros(n_steps)
    i_k_s_hist = np.zeros(n_steps)
    i_na_a_hist = np.zeros(n_steps)
    i_k_a_hist = np.zeros(n_steps)
    
    # Loop
    for i in range(n_steps):
        # 1. Update Gates FIRST (Semi-implicit or just standard Euler order)
        # Actually usually current is Calc -> Voltage Update -> Gate Update
        # Let's stick to the demo order: Currents -> Voltage -> Gates
        
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
        dVs = (i_stim_soma[i] - I_Na_s - I_K_s - I_L_s + I_axial_s) * inv_C_m
        dVa = (0.0            - I_Na_a - I_K_a - I_L_a + I_axial_a) * inv_C_m
        
        Vs += dVs * dt_ms
        Va += dVa * dt_ms
        
        # 3. Update Gates
        # Soma (Standard Kinetics)
        m_inf, tau_m, h_inf, tau_h, n_inf, tau_n = HHPhysics.get_gate_kinetics(Vs)
        ms += dt_ms * (m_inf - ms) / tau_m
        hs += dt_ms * (h_inf - hs) / tau_h
        ns += dt_ms * (n_inf - ns) / tau_n
        
        # AIS (Scaled Kinetics for Sharpening)
        m_inf, tau_m, h_inf, tau_h, n_inf, tau_n = HHPhysics.get_gate_kinetics(Va)
        ma += dt_ms * (m_inf - ma) / (tau_m * tau_m_scale_ais)
        ha += dt_ms * (h_inf - ha) / (tau_h * tau_h_scale_ais)
        na += dt_ms * (n_inf - na) / (tau_n * tau_n_scale_ais)
        
        # Store
        Vs_hist[i] = Vs
        Va_hist[i] = Va
        i_na_s_hist[i] = I_Na_s
        i_k_s_hist[i] = I_K_s
        i_na_a_hist[i] = I_Na_a
        i_k_a_hist[i] = I_K_a
        
    return t_ms, Vs_hist, Va_hist, i_na_s_hist, i_k_s_hist, i_na_a_hist, i_k_a_hist
