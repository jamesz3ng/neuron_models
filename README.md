## Quick Start

```bash
uv sync                              # install dependencies
python src/hh_cable.py                # run the HH cable equation
python src/ssds_model.py              # generate PCA basis from spike library
python benchmarks/bench_models.py hh_cable   # benchmark a model
```

Requires Python 3.10+. Dependencies: `numpy`, `matplotlib`, `scikit-learn`.

## Project Structure

```
neuron_models/
├── src/              # Core model implementations
├── benchmarks/       # Performance & fidelity benchmarks
├── demos/            # Visualisation and demo scripts
├── output/           # Generated plots (.png) and PCA basis data (.npz)
```

### `src/` — Models and Utilities

| File | Description |
|------|-------------|
| `physics.py` | `HHPhysics` class — HH constants, safe rate functions, steady-state calculations |
| `simulation.py` | Forward-Euler HH integrator, pulse train generation (dt=0.002 ms) |
| `analysis.py` | Spike detection (threshold crossing), waveform extraction |
| `hh_model.py` | HH point neuron|
| `hh_cable.py` | HH cable equation|
| `wave_model.py` | 1D wave equation for AP propagation |
| `cable_equation.py` | Passive cable equation (diffusion + decay, no active channels) |
| `event_model.py` | `EventPropagator` model: PCA encode/decode, multi-source convergence with LPF fusion, refractory filtering|
| `fast_model.py` | Delay-line model with absolute refractory period filtering |
| `ssds_model.py` | PCA basis generation pipeline — fatigue trains + population heterogeneity, exports `basis_data.npz` |
| `ais_simulation.py` | 2-compartment Soma-AIS model (high-density AIS channels, faster kinetics) |
| `simple_model.py` | Izhikevich neuron (standalone demo script) |
| `transport_model.py` | Transport equation for dendritic propagation (animated demo) |

### `benchmarks/` — Performance and Fidelity

| File | What it measures |
|------|-----------------|
| `bench_models.py` | CLI entry point — wall-clock timing with warmup/repeats for any model |
| `bench_spatial_complexity.py` | O(N) vs O(1) spatial scaling across 50–50,000 compartments(used in the abstract)|
| `bench_axon_length.py` | Axon length scaling, fixed resolution |
| `bench_scaling.py` | Neuron count scaling (1–5000 neurons) across 5 model types |
| `bench_speedup_comparison.py` | Grouped bar chart of slowdown factors relative to event model |
| `bench_bar_chart.py` | Conduction speed comparison (passive cable vs event/PCA) |
| `bench_pca_fidelity.py` | PCA reconstruction quality — RMSE, peak error, FWHM error, AHP error |
| `benchmark_shapes.py` | AP shape comparison (HH cable vs other models) |
| `benchmark_memory.py` | Memory usage across storage strategies (dense, sparse, boolean) |

### `demos/` — Visualisation and Exploration

Demos are standalone scripts that produce plots. Key ones:

- `demo_chain.py` — Coupled neuron chain via gap junctions with EventPropagator
- `demo_soma_ais.py` — Soma vs AIS spike shape validation
- `demo_refractory.py` / `demo_refractory_recovery.py` — Refractory period behavior
- `demo_event_convergence.py` — Multi-source convergence (LPF vs last-event fusion)
- `demo_fatigue.py` — Spike fatigue progression at 50/90/100 Hz
- `demo_dual_stim.py` — Dual stimulation comparison (Soma vs AIS vs Both)
- `debug_hh_physics.py` — Frequency sweep for adaptation analysis
- `sweep_ais_params.py` — AIS parameter sweep for sharp-AP parameters


### `output/`

Generated artifacts. Contains benchmark/demo plot PNGs and `basis_data.npz` (PCA mean waveform, 3 principal components, explained variance ratios, window parameters).