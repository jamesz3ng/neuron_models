#!/usr/bin/env python3
"""
Main interface for neuron models package.
Provides easy access to all model implementations.
"""

# Import from the parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hh_model_refactored import HodgkinHuxleyModel
from neuron_utils import SimulationParameters, NeuronConstants

# Import other models (to be added as they're refactored)
# from hh_cable_refactored import HodgkinHuxleyCableModel
# from wave_model_refactored import WaveModel

__all__ = [
    'HodgkinHuxleyModel',
    'SimulationParameters', 
    'NeuronConstants'
]


def get_available_models() -> dict:
    """
    Get dictionary of available neuron models.
    
    Returns:
        Dictionary mapping model names to model classes
    """
    return {
        'hh': HodgkinHuxleyModel,
        # 'hh_cable': HodgkinHuxleyCableModel,
        # 'wave': WaveModel
    }


def create_model(model_name: str, params: SimulationParameters = None):
    """
    Factory function to create neuron models.
    
    Args:
        model_name: Name of the model ('hh', 'hh_cable', 'wave')
        params: Simulation parameters (optional)
        
    Returns:
        Instantiated model object
        
    Raises:
        ValueError: If model_name is not recognized
    """
    available_models = get_available_models()
    
    if model_name not in available_models:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(available_models.keys())}")
    
    return available_models[model_name](params)