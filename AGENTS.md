# AGENTS.md

## The Problem
Our lab utilizes **CellML** to model biological phenomena. In CellML, neuronal models are defined using systems of Ordinary and Partial Differential Equations (ODEs/PDEs). While accurate, these models become a bottleneck at scale:
1. **HH Cable Equations** require high spatial discretization and multiple state variables per node, leading to slow simulation times.
2. **Standard Point-Neurons** (with fixed delays) are fast but fail to capture the morphological and temporal nuances of the action potential (AP) shape and propagation dynamics.

## The Approach
We are implementing and benchmarking a **1D Wave Equation** as a candidate for fast action potential propagation.
- **Baseline (Gold Standard):** Hodgkin-Huxley Cable Equation.
- **Proposed Model:** 1D Wave Equation (optimising for speed and constant velocity).
- **Core Requirement:** The reduced model must preserve the "functional shape" of the action potential (Width/FWHM, Peak Amplitude, and Rise/Fall times) to ensure biological relevance for downstream synaptic integration.

## Current Objectives
1. **Benchmarking:** Quantify the speedup of the Wave Equation vs. the HH Cable Equation.
2. **Validation:** Measure shape fidelity using RMSE, FWHM ratios, and peak alignment.
3. **CellML Integration:** Ensure the final optimized model can be expressed as a CellML-compatible system of ODEs.
4. **Refractory Dynamics (Planned):** Introduce a minimal recovery mechanism into the wave model to simulate absolute and relative refractory periods.

## Build & Run Commands
- **Install deps**: `uv sync`
- **Run model**: `python <model>.py` (e.g., `python hh_model.py`)
- **Benchmark**: `python bench_models.py <model>` (options: `hh_cable`, `hh_model`, `wave_model`, `common`)
- **No test/lint configured** - run scripts directly to verify

## Code Style Guidelines
- **Python**: 3.10+ required; use modern union syntax (`float | None`, not `Optional[float]`)
- **Imports**: stdlib first, third-party (`numpy as np`, `matplotlib.pyplot as plt`) second, local last
- **Naming**: `snake_case` for functions/variables; `PascalCase` for classes; physics params use domain notation (`C_m`, `g_Na`)
- **Private**: Prefix with underscore (`_default_params`, `_time_call`)
- **Type hints**: Required on function parameters; use keyword-only args (`*`) for multi-param functions
- **Defaults**: Use `_default_params()` factory pattern; document units in docstring
- **Returns**: Simulation functions return dicts with data + metadata
- **Errors**: Use `raise ValueError`/`SystemExit` for invalid inputs; add epsilon (`+ 1e-9`) for numerical stability
- **Main guard**: Always include `if __name__ == "__main__": main()`
- **Lazy imports**: Import `matplotlib` inside `main()` when module is also used as library
