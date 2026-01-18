"""
Core neuron model implementations.

Models:
- hh_model: Hodgkin-Huxley point neuron model
- hh_cable: Hodgkin-Huxley cable equation (spatial)
- wave_model: 1D wave equation for fast propagation
- cable_equation: Passive cable equation
- event_model: Event-based propagation with PCA
- fast_model: Delay-line model with refractory filtering
- simple_model: Izhikevich neuron model
- ssds_model: Spike Shape Decomposition System
- transport_model: Transport equation for dendrites

Core utilities:
- physics: HH physics constants and rate functions
- simulation: Simulation utilities and pulse generation
- analysis: Spike detection and analysis functions
"""

from .physics import HHPhysics
from .simulation import run_simulation, create_pulse_train, DT_MS
from .analysis import find_spike_peaks, measure_spike_width, extract_aligned_spike

from .hh_model import simulate_hh_model
from .hh_cable import simulate_hh_cable
from .wave_model import simulate_wave_model
from .cable_equation import simulate_cable_equation
from .event_model import EventPropagator
from .fast_model import simulate_fast_model

__all__ = [
    # Core utilities
    "HHPhysics",
    "run_simulation",
    "create_pulse_train",
    "DT_MS",
    "find_spike_peaks",
    "measure_spike_width",
    "extract_aligned_spike",
    # Models
    "simulate_hh_model",
    "simulate_hh_cable",
    "simulate_wave_model",
    "simulate_cable_equation",
    "EventPropagator",
    "simulate_fast_model",
]
