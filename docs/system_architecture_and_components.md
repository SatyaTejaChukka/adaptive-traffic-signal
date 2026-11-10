# System Architecture and Component Specification (`components.md`)

**Project Repository:** `SatyaTejaChukka/adaptive-traffic-signal`  
**Architecture Pattern:** 3-Pillar Physics-Informed Vector RL with Exact Counterfactual Introspection  

---

## 1. End-to-End System Architecture Diagram

```
+---------------------------------------------------------------------------------------+
|                        PAIRED MULTI-SEED EXPERIMENT RUNNER                            |
|     Scenarios: S1 (2W Surge), S2 (Balanced Mixed), S3 (Freight), S4 (Regime Shift)    |
|     CRN Streams: `arrival_rng = Random(seed)` | `agent_rng = Random(seed + 10000)`    |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
|                     PHYSICAL HETEROGENEOUS TRAFFIC SIMULATOR                          |
|                                  (simulation.py)                                      |
|  - Tracks per-approach queues Q_d, vehicle classes, wait times W_d, and red time t_red|
|  - Executes physical seepage burst + HV startup lag + saturation discharge            |
+---------------------+-------------------------------------------+---------------------+
                      | (Q_d, t_red_d, W_d)                       ^ (Executed Action a*)
                      v                                           |
+-------------------------------------------+                     |
| PILLAR 1: SEEPAGE DISCHARGE ESTIMATOR     |                     |
|              (rl_state.py)                |                     |
| - Computes pore-blockage rho_HV           |                     |
| - Computes red-phase seepage alpha_seep   |                     |
| - Outputs T_clear(NS), T_clear(EW) in sec |                     |
| - Quantizes to 6-tuple discrete state s_t |                     |
+---------------------+---------------------+                     |
                      | (State s_t, T_clear, W_d)                 |
                      v                                           |
+-------------------------------------------+                     |
| PILLAR 2: CONSTRAINED VECTOR Q-LEARNER    |---------------------+
|        (rl_agent.py & reward.py)          |
| - 4-Profile Vector Q-Bank Q_m(s,a) in R^3 |
| - Hierarchical Starvation Mask M(s,a)     |
| - Zero-Retraining C-GPI Action Selection  |
| - Parallel SMDP Vector Bellman Updates    |
+---------------------+---------------------+
                      | (Q_m(s,.), M(s,.), w_active, T_clear)
                      v
+---------------------------------------------------------------------------------------+
| PILLAR 3: DETERMINISTIC COUNTERFACTUAL BOUNDARY INTROSPECTOR                          |
|                         (counterfactual_explainer.py)                                 |
|  - Solves w^T (Q(s, a_1*) - Q(s, a_2*)) = 0 for exact 3-head percentage attribution   |
|  - Evaluates instantaneous Profile-Flip Sensitivity across all canonical profiles     |
|  - Performs 1D State-Lattice Walk + T_clear Inversion for physical truck/wait deltas  |
+---------------------------------------------------------------------------------------+
```

---

## 2. Module Responsibilities & File Mapping

| File Path | Pillar | Primary Responsibility |
| :--- | :---: | :--- |
| `rl_state.py` | **Pillar 1** | Implements `SeepageDischargeModel` (`compute_seepage_clear_time`), action-aligned clearance bucketization (`discretize_clear_time`), waiting-time bucketization (`discretize_wait_time`), and `build_seepage_rl_state()`. |
| `rl_agent.py` | **Pillar 2** | Implements `DecomposedVectorQLearner`, 3-component SMDP vector reward calculation (`compute_smdp_vector_reward`), hierarchical starvation action masking (`compute_starvation_mask`), C-GPI action selection (`select_action`), and parallel multi-profile Bellman backups (`update`). |
| `counterfactual_explainer.py` | **Pillar 3** | Implements `CounterfactualIntrospector`, solving linear attribution, profile-flip boundaries, Simplex minimum weight-perturbation norms, and physical vehicle/waiting tipping points (`explain_decision`). |
| `tests/test_three_pillar_core.py` | **Verification** | Automated `pytest` suite verifying two-wheeler conservation across $\alpha_{\text{seep}} \in [0,1]$, monotonic HV lag, zero-retraining profile flips, deadlock-free dual-starvation masking, and exact counterfactual math. |

---

## 3. State-Space, Action-Space, and Memory Footprint Specifications

### 3.1 Discrete State Tuple ($s_t \in \mathcal{S}$)
$$s_t = \left( b_{T,\text{NS}}, \, b_{T,\text{EW}}, \, \phi_t, \, y_t, \, b_{W,\text{NS}}, \, b_{W,\text{EW}} \right)$$
* $b_{T,\text{NS}}, b_{T,\text{EW}} \in \{0, 1, 2, 3\}$: Discretized clearance time $T_{\text{clear}}$ buckets with cutoffs $(4.0\text{ s}, 8.5\text{ s}, 13.5\text{ s})$ aligned to $\{5\text{ s}, 10\text{ s}, 15\text{ s}\}$ green durations.
* $\phi_t \in \{0, 1\}$: Active signal phase ($0 = \text{NS}$, $1 = \text{EW}$).
* $y_t \in \{0, 1\}$: Yellow transition flag ($0 = \text{Green}$, $1 = \text{Yellow}$).
* $b_{W,\text{NS}}, b_{W,\text{EW}} \in \{0, 1, 2\}$: Discretized maximum waiting time buckets with cutoffs $(15.0\text{ s}, 40.0\text{ s})$.
* **Cardinality:** $|\mathcal{S}| = 4 \times 4 \times 2 \times 2 \times 3 \times 3 = 576$ states.

### 3.2 Action Space ($\mathcal{A}$)
* $|\mathcal{A}| = 6$ discrete phase-duration macro-actions:
  * `0`: $(\text{NS}, 5\text{ s})$, `1`: $(\text{NS}, 10\text{ s})$, `2`: $(\text{NS}, 15\text{ s})$
  * `3`: $(\text{EW}, 5\text{ s})$, `4`: $(\text{EW}, 10\text{ s})$, `5`: $(\text{EW}, 15\text{ s})$

### 3.3 Vector Q-Bank Memory Footprint
* $M = 4$ profiles $\times 576$ states $\times 6$ actions $\times 3$ reward heads $= 41,472$ `float64` values ($\approx 331.8\text{ KB}$ total memory), enabling sub-millisecond inference on embedded edge controllers (e.g., Raspberry Pi / ARM Cortex).

---

## 4. Ablation Matrix for Experimental Validation

| Model ID | Demand / State Representation | Value Function & Policy | Fairness Mechanism | Role in Paper |
| :--- | :--- | :--- | :--- | :--- |
| **`M0_COUNT`** | Raw vehicle count $N_d$ | Scalar $Q(s,a)$ | External heuristic override | Legacy baseline in repo |
| **`M1_STATIC_PCU`** | Linear static PCU sum | Scalar $Q(s,a)$ | External heuristic override | Standard heterogeneous RL baseline |
| **`M2_SEEPAGE_SCALAR`** | Physics-informed $T_{\text{clear}}(d)$ | Scalar $Q(s,a)$ | External heuristic override | Isolates Pillar 1 ($T_{\text{clear}}$ gain) |
| **`M3_VECTOR_UNMASKED`** | Physics-informed $T_{\text{clear}}(d)$ | 4-Profile Vector $\mathbf{Q}_m(s,a)$ + C-GPI | Soft reward penalty only | Isolates Pillar 2 (zero-retraining shift) |
| **`M4_FULL_PROPOSED`** | Physics-informed $T_{\text{clear}}(d)$ | 4-Profile Vector $\mathbf{Q}_m(s,a)$ + C-GPI | Internal Mask $\mathcal{M}(s,a)$ + EQPI | **Full Proposed System (Pillars 1+2+3)** |