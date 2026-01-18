
import numpy as np

class HHPhysics:
    """
    Hodgkin-Huxley model physics with safe rate functions.
    Encapsulates all Hodgkin-Huxley constants and functions.
    """

    # Membrane capacitance and conductances
    C_m = 1.0  # μF/cm²
    g_Na = 120.0  # mS/cm²
    g_K = 36.0
    g_L = 0.3
    
    # Reversal potentials (mV)
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387
    v_rest = -65.0

    # Singularity voltages for vtrap
    _V_SING_M = -40.0  # α_m singularity
    _V_SING_N = -55.0  # α_n singularity
    _VTRAP_EPS = 1e-7  # Threshold for vtrap activation

    @staticmethod
    def _vtrap(x: float, y: float) -> float:
        """
        Safe evaluation of x/(exp(x/y) - 1) near singularities.

        Uses first-order Taylor expansion when |x| < epsilon:
        x/(exp(x/y) - 1) ≈ y - x/2 + O(x²)
        """
        if abs(x) < HHPhysics._VTRAP_EPS:
            return y - x / 2.0
        return x / (np.exp(x / y) - 1.0)

    @staticmethod
    def rates(V: float) -> tuple[float, float, float, float, float, float]:
        """
        Compute HH rate constants at voltage V.

        Returns (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n).
        Uses vtrap for α_m (at -40mV) and α_n (at -55mV) to avoid division by zero.
        """
        # α_m: vtrap at V = -40mV
        x_m = V + 40.0
        alpha_m = -0.1 * HHPhysics._vtrap(x_m, -10.0)
        beta_m = 4.0 * np.exp((V + 65.0) / -18.0)

        # α_h, β_h: no singularities
        alpha_h = 0.07 * np.exp((V + 65.0) / -20.0)
        beta_h = 1.0 / (1.0 + np.exp((V + 35.0) / -10.0))

        # α_n: vtrap at V = -55mV
        x_n = V + 55.0
        alpha_n = -0.01 * HHPhysics._vtrap(x_n, -10.0)
        beta_n = 0.125 * np.exp((V + 65.0) / -80.0)

        return alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n

    @staticmethod
    def steady_state(V: float) -> tuple[float, float, float]:
        """
        Compute steady-state gate values at voltage V.

        Returns (m_inf, h_inf, n_inf).
        """
        alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = HHPhysics.rates(V)

        m_inf = alpha_m / (alpha_m + beta_m)
        h_inf = alpha_h / (alpha_h + beta_h)
        n_inf = alpha_n / (alpha_n + beta_n)

        return m_inf, h_inf, n_inf
