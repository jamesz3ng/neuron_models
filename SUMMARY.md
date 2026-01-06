# Spike Shape Decomposition System (SSDS) - Project Summary

## Overview

This project implements a **Spike Shape Decomposition System (SSDS)** that uses Principal Component Analysis (PCA) to create a compact, biologically-faithful representation of action potential (AP) waveforms. The goal is to enable fast neuron simulations that preserve the "functional shape" of action potentials for downstream synaptic integration.

---

## The Problem

Our lab uses **CellML** to model biological phenomena. Neuronal models defined using ODEs/PDEs become computational bottlenecks at scale:

| Model Type | Pros | Cons |
|------------|------|------|
| **HH Cable Equations** | Biologically accurate | Slow (high spatial discretization, multiple state variables) |
| **Point Neurons (fixed delay)** | Fast | Loses AP shape and propagation dynamics |

**Our Solution**: Create a reduced model that is fast but preserves the biologically-relevant spike shape features (Width/FWHM, Peak Amplitude, Rise/Fall times).

---

## What We Built

### 1. Spike Shape Decomposition System (`ssds_model.py`)

A comprehensive system that:
- Generates diverse action potential waveforms using the Hodgkin-Huxley model
- Performs PCA to extract principal components ("eigen-spikes")
- Validates reconstruction quality across extreme conditions
- Exports basis functions for use in fast simulations

#### Four Spike Generation Protocols

| Protocol | Purpose | Implementation | Spikes Generated |
|----------|---------|----------------|------------------|
| **A: Fatigue Spectrum** | Capture spike frequency adaptation | 10-pulse trains at [75, 80, 85] Hz | ~30 |
| **B: Refractory Curve** | Fine-grained recovery dynamics | Paired pulses, ISI 2.0-20.0ms (0.2ms steps) | ~56 |
| **C: Population Heterogeneity** | Biological diversity | g_Na/g_K ±15% variation, 1000 samples | 1000 |
| **D: Hyperpolarization Rebound** | "Super-charged" tall/sharp spikes | V_init from -70 to -90mV | 50 |

**Total: ~1,136 spikes** covering the full continuous space of AP shapes.

### 2. HH Physics Debugging Tool (`debug_hh_physics.py`)

Diagnostic tool that:
- Sweeps stimulation frequencies to find spike frequency adaptation
- Identifies the "sweet spot" frequency for meaningful adaptation without spike failure
- Detects depolarization block conditions
- Generates `hh_stress_test.png` visualization

**Key Finding**: The standard HH model has a ~10ms refractory period. Sweet spot is **75Hz** (all spikes fire with ~4% adaptation). At 80-85Hz, stronger adaptation (-11%) occurs with some spike loss.

### 3. Exported Basis Functions (`basis_data.npz`)

Compact representation of spike shape space:

```python
{
    'mean_waveform': (1000,),      # Mean spike shape
    'components': (3, 1000),       # Top 3 principal components
    'explained_variance': (3,),    # [0.92, 0.03, 0.03]
    'dt_ms': 0.01,                 # Time step
    'window_pre_ms': 2.0,          # Window before peak
    'window_post_ms': 8.0,         # Window after peak
    'window_samples': 1000         # Total samples
}
```

---

## Key Results

### PCA Performance

| Metric | Value |
|--------|-------|
| PC1 Variance | 92.2% |
| PC2 Variance | 2.9% |
| PC3 Variance | 2.6% |
| **Total (3 PCs)** | **97.7%** |

### Reconstruction Quality

| Condition | RMSE (2 PCs) | RMSE (3 PCs) |
|-----------|--------------|--------------|
| Mean (all spikes) | 0.37 mV | 0.24 mV |
| 95th percentile | 1.10 mV | - |
| 99th percentile | 1.61 mV | - |
| Fatigued 85Hz | 1.06 mV | 0.85 mV |
| Super-charged -90mV | 1.52 mV | 1.30 mV |

### Spike Amplitude Range (Balanced)

| Spike Type | Peak Voltage | Change from Rest |
|------------|--------------|------------------|
| Fatigued (85Hz, last spike) | ~36 mV | **-11%** |
| Standard (rest) | ~40.6 mV | baseline |
| Super-charged (-90mV hyperpol) | ~48.2 mV | **+19%** |

The basis can now reconstruct spikes ranging from **-11% to +19%** amplitude variation.

---

## File Structure

```
/neuron_models/
├── ssds_model.py          # Main SSDS implementation
├── debug_hh_physics.py    # HH frequency sweep diagnostics
├── hh_model.py            # Hodgkin-Huxley point neuron
├── hh_cable.py            # HH cable equation (spatial)
├── fast_model.py          # Delay-line with refractory filtering
├── wave_model.py          # 1D wave equation model
├── cable_equation.py      # Passive cable equation
├── bench_models.py        # Model benchmarking
├── bench_scaling.py       # Speed scaling analysis
├── benchmark_shapes.py    # AP shape comparison
├── benchmark_memory.py    # Memory usage comparison
├── demo_refractory.py     # Refractory period demo
│
├── basis_data.npz         # ⭐ Exported PCA basis functions
├── spike_pca_analysis.png # Spike library + eigen-spikes visualization
├── spike_validation.png   # Multi-panel reconstruction validation
├── hh_stress_test.png     # Frequency sweep results
│
├── AGENTS.md              # Coding guidelines
├── SUMMARY.md             # This file
└── README.md              # Project overview
```

---

## Technical Details

### Spike Extraction Parameters

```python
DT_MS = 0.01              # 0.01ms time step (100kHz sampling)
WINDOW_PRE_MS = 2.0       # 2ms before peak
WINDOW_POST_MS = 8.0      # 8ms after peak
WINDOW_POINTS = 1000      # Total points per spike
SPIKE_THRESHOLD_MV = -20.0
STIM_AMPLITUDE = 30.0     # nA/mm² (sufficient for reliable spiking)
```

### HH Model Steady-State Gate Values

| Voltage | h_inf (Na inactivation) | Effect |
|---------|-------------------------|--------|
| -65 mV (rest) | 0.596 | Standard spike |
| -80 mV | 0.931 | Taller spike |
| -90 mV | 0.984 | Maximum "super-charged" spike |

### Protocol D: Hyperpolarization Physics

When a neuron is held at hyperpolarized potentials:
1. Sodium inactivation gate (h) de-inactivates: h → ~1.0
2. More Na+ channels available for activation
3. Resulting spike is **taller** and **sharper**
4. Balances the "fatigued" spikes in PCA space

---

## What's Next

### Immediate Tasks

1. **CellML Integration**
   - Express the PCA-based spike model as CellML-compatible ODEs
   - The reconstruction formula: `V(t) = mean(t) + w1*PC1(t) + w2*PC2(t) + w3*PC3(t)`
   - Need ODEs to compute weights (w1, w2, w3) based on neuron state

2. **Weight Prediction Model**
   - Train a mapping from neuron state → PCA weights
   - Inputs: firing history, membrane potential, time since last spike
   - Outputs: (w1, w2, w3) for spike shape reconstruction

3. **Integration with `fast_model.py`**
   - Replace fixed spike template with PCA-reconstructed waveforms
   - Use refractory period to modulate weights (fatigued spikes)
   - Use pre-spike voltage to modulate weights (super-charged spikes)

### Future Enhancements

1. **Slow Inactivation**
   - Add slow Na+ inactivation to HH model for longer-term adaptation
   - Would increase fatigue effects beyond current -11%

2. **Cable Equation Integration**
   - Apply SSDS to spatial propagation (HH cable)
   - Verify shape preservation during propagation

3. **Validation Against Experimental Data**
   - Compare PCA components to recorded intracellular AP shapes
   - Validate that eigen-spikes match known biophysical phenomena

4. **Performance Benchmarking**
   - Quantify speedup of PCA-based reconstruction vs. full HH integration
   - Target: 10-100x speedup while maintaining <1mV shape error

---

## Usage

### Generate Basis Functions

```bash
python ssds_model.py
```

Outputs:
- `basis_data.npz` - PCA basis for spike reconstruction
- `spike_pca_analysis.png` - Visualization of spike library
- `spike_validation.png` - Reconstruction quality validation

### Debug HH Physics

```bash
python debug_hh_physics.py
```

Outputs:
- `hh_stress_test.png` - Frequency sweep analysis
- Console table of adaptation metrics

### Load Basis Functions

```python
import numpy as np

data = np.load('basis_data.npz')
mean = data['mean_waveform']       # (1000,)
components = data['components']    # (3, 1000)
dt_ms = float(data['dt_ms'])       # 0.01

# Reconstruct a spike with weights
def reconstruct(w1, w2, w3):
    return mean + w1*components[0] + w2*components[1] + w3*components[2]

# Example: fatigued spike (negative w1)
fatigued_spike = reconstruct(w1=-5.0, w2=1.0, w3=0.5)

# Example: super-charged spike (positive w1)
supercharged_spike = reconstruct(w1=8.0, w2=-1.0, w3=0.5)
```

---

## Key Insights

1. **2-3 PCs capture 95-98% of spike shape variance** across all physiological conditions

2. **PC1 (~92%)** captures overall spike amplitude/duration scaling

3. **PC2 (~3%)** captures amplitude modulation (grow/shrink)

4. **PC3 (~3%)** captures asymmetric shape changes (rise vs. fall dynamics)

5. **The HH model's refractory period (~10ms) limits sustainable firing to ~100Hz**

6. **Hyperpolarization rebound provides +19% amplitude boost** (important for post-inhibitory rebound spiking)

7. **Compact representation**: 3000 floats (mean + 3 PCs) + 3 weights per spike can reconstruct any physiological AP shape with <1mV error for 95% of cases

---

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve.
- CellML: https://www.cellml.org/
- Project Repository: `/Users/jameszeng/Documents/neuron_models/`

---

*Last updated: January 2025*
