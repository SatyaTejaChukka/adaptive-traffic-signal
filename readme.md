# SeepageRL: Physics-Informed Zero-Retraining Multi-Objective Traffic Signal Control with Exact Counterfactual Auditability

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Verification Suite](https://img.shields.io/badge/pytest-5%20passed-brightgreen.svg)](tests/test_three_pillar_core.py)

**Target Publication:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Transportation Research Part C / IEEE ITSC  
**Architecture:** 3-Pillar Heterogeneous Traffic Control (Physics Seepage Model + Vector Q-Learning with C-GPI + Deterministic Counterfactual Introspection)

---

## 1. Overview & Research Highlights

Standard reinforcement learning (RL) traffic controllers make two invalid assumptions when applied to heterogeneous, non-lane-disciplined traffic (e.g., South Asian and Indian cities):
1. **The Linear Discharge Fallacy:** Two-wheelers filter laterally to the stop-line during red phases (**seepage**) discharging in a high-density burst ($S_{\text{burst}} = 3.6\text{ veh/s}$), while heavy vehicles block lateral channels and introduce startup lag ($\tau_{\text{lag}}$). Treating vehicles as static PCU counts severely degrades performance.
2. **Retraining Lock-In:** Traditional scalar RL models require complete retraining when operational priorities change (e.g., morning commuter rush $\to$ industrial freight logistics).

**SeepageRL solves these issues through a unified 3-Pillar Architecture:**
- **Pillar 1 (Seepage Discharge Physics):** Derives continuous clearance time $T_{\text{clear}}(d)$ using a 3-stage discharge equation (Burst + HV Startup Lag + Main Platoon Saturation Flow) while strictly conserving vehicles across all seepage rates.
- **Pillar 2 (Constrained Vector Q-Learning with C-GPI):** Maintains a multi-profile vector Q-bank across 4 canonical profiles (`balanced`, `commuter_priority`, `freight_priority`, `emergency_corridor`). Uses Constrained Generalized Policy Improvement (C-GPI) to switch operational profiles in $<0.2\text{ ms}$ with **zero retraining**, guarded by a Hierarchical Starvation Action Mask $\mathcal{M}(s,a)$.
- **Pillar 3 (Deterministic Counterfactual Auditability - EQPI):** Analytically solves the decision boundary $\mathbf{w}^\top (\mathbf{Q}(s, a_1^*) - \mathbf{Q}(s, a_2^*)) = 0$ to output exact component attribution (Throughput / Delay / Priority) and physical sensor tipping points (e.g., *"+2 Trucks on EW flips signal"*).

---

## 2. Interactive Simulation & Research Cockpit

The interactive simulator runs a **widescreen 1800 × 800 layout** with a dedicated 400px live telemetry cockpit:

```bash
python simulation.py
```

### Visual Enhancements
- **Stop-Line Bike Boxes:** Highlights lateral seepage reservoirs where motorcycles aggregate during red phases.
- **Pillar 1 Gauge Card:** Real-time $T_{\text{clear}}$, $\alpha_{\text{seep}}$ (%), and Heavy Vehicle Pore-Blockage $\rho_{\text{HV}}$ (%).
- **Pillar 3 XAI Card:** Decision margin $\Delta Q$, tripartite attribution stacked bar, and physical tipping-point sensor readouts.
- **Approach Wait Auroras:** Color-coded warning borders approaching the starvation threshold ($W_{\max} = 45\text{ s}$).

### Real-Time Interactive Hotkeys
| Key | Function | Description |
| :---: | :--- | :--- |
| **`1`** | **Balanced Profile** | Switches to equal throughput and delay fairness $\mathbf{w} = (0.40, 0.40, 0.20)$ |
| **`2`** | **Commuter Priority** | Prioritizes passenger cars & motorcycles $\mathbf{w} = (0.55, 0.35, 0.10)$ |
| **`3`** | **Freight Priority** | Prioritizes heavy trucks & buses $\mathbf{w} = (0.20, 0.25, 0.55)$ |
| **`4`** | **Emergency Corridor** | Maximum priority preemption $\mathbf{w} = (0.10, 0.15, 0.75)$ |
| **`M`** | **Toggle Controller** | Seamlessly toggle between **RL (C-GPI)** and **Fixed-Timer Baseline** |
| **`S`** | **Toggle Starvation Mask** | Enables/disables hard constraint mask $\mathcal{M}(s,a)$ live |
| **`Space`** | **Pause / Resume** | Halts simulation to inspect active Q-values and queues |
| **`+` / `-`** | **Playback Speed** | Adjusts speed: $0.5\times$, $1\times$, $2\times$, $4\times$ |
| **`B`** | **Inject Truck Wave** | Spawns a surge of 5 heavy trucks/buses on East-West corridor |
| **`T`** | **Inject 2W Swarm** | Spawns a swarm of 12 motorcycles on North-South corridor |

---

## 3. High-Speed Headless Benchmarking & Experimentation

The codebase includes a decoupled discrete-event simulation engine (`headless_runner.py`) running at **$>250$ simulation episodes per second** with Common Random Numbers (CRN) for 100% reproducible scientific evaluation.

### Train Vector Q-Bank (1,000 Episodes)
```bash
python -c "from headless_runner import train_seepage_vector_agent; train_seepage_vector_agent(num_episodes=1000)"
```

### Run 200 Paired Ablation Benchmark (5 Models × 4 Scenarios × 10 Seeds)
```bash
python run_experiments.py
```
Outputs complete CSV logs with uncensored vehicle delays, 95th-percentile waiting time $W_{95}$, throughput, and starvation violations to `experiments/benchmark_results.csv`.

### Generate Publication Vector Figures
```bash
python plot_paper_figures.py
```
Generates IEEE publication-formatted figures in `images/paper_figures/`:
- `fig1_seepage_discharge_physics.png`: Verification of two-wheeler seepage acceleration and HV blockage suppression.
- `fig2_ablation_mean_delay.png`: Comparative delay reductions across Models M0–M4 and Scenarios S1–S4.
- `fig3_fairness_cdf_starvation.png`: Empirical CDF of maximum waiting time proving the hard $W_{\max} = 45\text{ s}$ bound.

---

## 4. Verification Test Suite

Run the automated mathematical invariant test suite:
```bash
python -m pytest tests/test_three_pillar_core.py -v
```

**Checked Invariants:**
- `test_vehicle_conservation_and_seepage_acceleration`: Validates vehicle conservation and strict delay reduction under front-seepage burst.
- `test_heavy_vehicle_pore_blockage_and_startup_lag`: Validates non-linear heavy-vehicle startup inertia and lateral pore blockage.
- `test_zero_retraining_profile_switch_flips_optimal_action`: Validates C-GPI instant action switching in $<0.2\text{ ms}$.
- `test_dual_starvation_mask_never_deadlocks`: Proves deadlock-free operation when both approaches exceed $W_{\max}$.
- `test_counterfactual_attribution_and_physical_tipping_point`: Verifies exact $100\%$ component attribution and integer vehicle tipping points.

---

## 5. Repository Structure

```
├── config.py                 # Central configurations, Simplex weights, seepage constants
├── rl_state.py               # Pillar 1: Seepage queue discharge & state encoding
├── rl_agent.py               # Pillar 2: DecomposedVectorQLearner + C-GPI + Starvation Mask
├── counterfactual_explainer.py # Pillar 3: Deterministic counterfactual boundary introspector
├── reward.py                 # Duration-normalized SMDP vector reward functions
├── simulation.py             # 1800x800 interactive UI with live research telemetry cockpit
├── headless_runner.py        # High-speed headless simulation engine with dual-RNG CRN
├── run_experiments.py        # Paired ablation benchmark runner (200 runs to CSV)
├── plot_paper_figures.py     # Publication figure generator
├── tests/
│   └── test_three_pillar_core.py # Comprehensive pytest verification suite
├── experiments/
│   └── benchmark_results.csv # Empirical experimental results
└── images/
    └── paper_figures/        # Generated IEEE research figures
```
