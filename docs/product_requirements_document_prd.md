# Product Requirements Document (PRD): Zero-Retraining Multi-Objective Traffic Signal Control for Heterogeneous Seepage Traffic

**Project Repository:** `SatyaTejaChukka/adaptive-traffic-signal`  
**Document Version:** 2.0.0 (3-Pillar Unified Architecture)  
**Proposed Paper Title:** *Zero-Retraining Multi-Objective Traffic Signal Control for Heterogeneous Seepage Traffic via Reward-Decomposed Q-Learning and Counterfactual Action Masking*  
**Target Venues:** IEEE Transactions on Intelligent Transportation Systems (T-ITS) / Transportation Research Part C  

---

## 1. Executive Summary & Scientific Motivation

Existing reinforcement learning (RL) traffic signal controllers suffer from three structural limitations when deployed in heterogeneous, non-lane-disciplined Indian traffic:
1. **Linear Queue Discharge Fallacy:** Standard count-based and static Passenger Car Unit (PCU) representations assume queues discharge linearly. In reality, two-wheelers (`bike`) and three-wheelers (`rickshaw`) filter through lateral pore spaces to the stop line during red phases (**seepage**) and discharge in a high-density initial burst, whereas heavy vehicles (`bus`, `truck`) impose non-linear startup inertia lag and block lateral seepage channels.
2. **Retraining Lock-In & Unlearned Starvation Overrides:** Switching between operational traffic regimes (`balanced`, `commuter_priority`, `freight_priority`, `emergency_corridor`) in scalar Q-learning invalidates the learned Q-table because scalar rewards entangle throughput, delay, and class priority into a single number. Furthermore, applying an external heuristic starvation override corrupts off-policy Bellman updates and lacks a formal worst-case waiting bound during dual-approach gridlock.
3. **Opaque Black-Box Decisions:** Deep explainable RL (XRL) controllers produce soft attention heatmaps that cannot tell a traffic engineer the exact mathematical tipping point (e.g., how many additional heavy trucks or seconds of waiting time on the opposing approach would flip the signal decision).

This PRD defines a unified, lightweight, CPU-executable 3-pillar architecture that upgrades the `adaptive-traffic-signal` repository without requiring deep neural networks, GPUs, or external simulators.

---

## 2. The Three Core Technical Contributions

### 2.1 Pillar 1: Physics-Informed Seepage-Aware Queue Discharge Model ($T_{\text{clear}}$)
For each approach group $d \in \{\text{NS}, \text{EW}\}$ with waiting vehicle counts $N_{k,d}$ across classes $k \in \{\text{bike}, \text{rickshaw}, \text{car}, \text{bus}, \text{truck}\}$ and elapsed red time $t_{\text{red},d}$, we replace static linear weights with a three-stage physical clearance time estimator $T_{\text{clear}}(d)$ (in seconds).

First, the heavy-vehicle pore-blockage factor $\rho_{\text{HV},d} \in [0, 1]$ and active red-phase seepage fraction $\alpha_{\text{seep}}(t_{\text{red},d}, \rho_{\text{HV},d}) \in [0, \alpha_{\max}]$ are computed as:

$$\rho_{\text{HV},d} = \min\left(1.0, \, \frac{N_{\text{bus},d} + N_{\text{truck},d}}{\max(1, N_{\text{total},d})} \cdot \beta_{\text{block}}\right)$$

$$\alpha_{\text{seep}}(t_{\text{red},d}, \rho_{\text{HV},d}) = \alpha_{\max} \left(1 - e^{-t_{\text{red},d} / \tau_{\text{seep}}}\right) \left(1 - \rho_{\text{HV},d}\right)$$

Second, total queue clearance time $T_{\text{clear}}(d)$ partitions two-wheelers strictly between the front-seeped burst cohort and the non-seeped main platoon (preserving strict vehicle conservation across all $\alpha_{\text{seep}} \in [0, 1]$):

$$N_{\text{seep},d} = \alpha_{\text{seep}} \left(N_{\text{bike},d} + 0.5 N_{\text{rickshaw},d}\right)$$

$$\tau_{\text{lag}}(N_{\text{HV},d}) = \tau_{\max} \left(1 - e^{-\lambda_{\text{HV}} \left(N_{\text{bus},d} + N_{\text{truck},d}\right)}\right)$$

$$\text{PCU}_{\text{platoon},d} = (1 - \alpha_{\text{seep}}) N_{\text{bike},d} \text{PCU}_{\text{bike}} + (1 - 0.5\alpha_{\text{seep}}) N_{\text{rickshaw},d} \text{PCU}_{\text{rickshaw}} + \sum_{k \in \{\text{car, bus, truck}\}} N_{k,d} \text{PCU}_k$$

$$T_{\text{clear}}(d) = \underbrace{\frac{N_{\text{seep},d}}{S_{\text{burst}}}}_{\text{Stage 1: Front Seepage Burst}} + \underbrace{\tau_{\text{lag}}(N_{\text{HV},d})}_{\text{Stage 2: HV Startup Lag}} + \underbrace{\frac{\text{PCU}_{\text{platoon},d}}{S_{\text{sat}}}}_{\text{Stage 3: Main Platoon Saturation Flow}}$$

Because $S_{\text{burst}} \gt \frac{S_{\text{sat}}}{\text{PCU}_{\text{bike}}}$, higher red-phase seepage reduces required green time, and $T_{\text{clear}}(d)$ maps directly onto the controller's discrete green-duration actions $\{5\text{ s}, 10\text{ s}, 15\text{ s}\}$.

---

### 2.2 Pillar 2: Constrained Reward-Decomposed Vector Q-Learning with C-GPI
Instead of a scalar Q-table, the controller maintains a 3-dimensional vector reward $\mathbf{r}_t = [r_{\text{tp}}, r_{\text{wait}}, r_{\text{prio}}]^\top \in \mathbb{R}^3$ and a multi-profile vector Q-bank $\mathbf{Q}_m(s,a) \in \mathbb{R}^3$ across $M = 4$ canonical profiles (`balanced`, `commuter_priority`, `freight_priority`, `emergency_corridor`) with preference weight vectors $\mathbf{w}_m \in \Delta^2$:

| Profile Name ($m$) | Throughput Weight ($w_{\text{tp}}$) | Waiting Penalty Weight ($w_{\text{wait}}$) | Class Priority Weight ($w_{\text{prio}}$) | Operational Objective |
| :--- | :---: | :---: | :---: | :--- |
| `balanced` | $0.40$ | $0.40$ | $0.20$ | Default urban mixed-traffic operation |
| `commuter_priority` | $0.55$ | $0.35$ | $0.10$ | Peak two-wheeler and passenger car rush |
| `freight_priority` | $0.20$ | $0.25$ | $0.55$ | Industrial corridor heavy-vehicle progression |
| `emergency_corridor` | $0.10$ | $0.15$ | $0.75$ | High-priority transit and freight preemption |

#### Hierarchical Starvation Action Mask $\mathcal{M}(s,a)$
To enforce a hard upper bound $W_{\max}$ on waiting time without causing a dual-starvation deadlock when both approaches exceed $W_{\max}$, the action mask $\mathcal{M}(s,a) \in \{0, -\infty\}$ is defined as:

$$d_{\text{starved}} = \arg\max_{d \in \{\text{NS}, \text{EW}\}} W_d, \qquad g_{\min}(d) = \min_{g \in \{5, 10, 15\}} \left\{ g \;\middle|\; g \ge \min\left(15.0, \, T_{\text{clear}}(d)\right) \right\}$$

$$\mathcal{M}(s,a) = \begin{cases} 0 & \text{if } \max(W_{\text{NS}}, W_{\text{EW}}) \lt W_{\max} \\ 0 & \text{if } \max(W_{\text{NS}}, W_{\text{EW}}) \ge W_{\max} \text{ and } \text{phase}(a) = d_{\text{starved}} \text{ and } g(a) \ge g_{\min}(d_{\text{starved}}) \\ -\infty & \text{otherwise} \end{cases}$$

#### Constrained Generalized Policy Improvement (C-GPI) & SMDP Bellman Backup
At runtime, for any active preference vector $\mathbf{w}_{\text{runtime}} \in \Delta^2$, the controller selects the optimal action in under $1\text{ ms}$ with zero retraining via C-GPI:

$$a^* = \arg\max_{a \in \mathcal{A}} \left( \max_{m \in \{1,\dots,M\}} \mathbf{w}_{\text{runtime}}^\top \mathbf{Q}_m(s,a) + \mathcal{M}(s,a) \right)$$

On every macro-transition $(s, a_{\text{exec}}, \mathbf{r}_{\text{rate}}, s', \Delta t)$, all $M$ profile heads are updated synchronously in parallel using Semi-Markov continuous-time discounting $\gamma^{\Delta t / 5.0}$:

$$a'_m = \arg\max_{a' \in \mathcal{A}} \left( \mathbf{w}_m^\top \mathbf{Q}_m(s', a') + \mathcal{M}(s', a') \right)$$

$$\mathbf{Q}_m(s, a_{\text{exec}}) \leftarrow \mathbf{Q}_m(s, a_{\text{exec}}) + \alpha \left[ \mathbf{r}_{\text{rate}} + \gamma^{\Delta t / 5.0} \mathbf{Q}_m(s', a'_m) - \mathbf{Q}_m(s, a_{\text{exec}}) \right] \quad \forall m \in \{1,\dots,M\}$$

---

### 2.3 Pillar 3: Deterministic Counterfactual Boundary Introspection (EQPI)
Because the action-value scalarization is an exact linear inner product $\mathbf{w}^\top \mathbf{Q}(s,a)$, for every chosen action $a_1^*$ and runner-up action $a_2^*$, the controller solves the decision boundary $\mathbf{w}^\top \left(\mathbf{Q}(s, a_1^*) - \mathbf{Q}(s, a_2^*)\right) = 0$ analytically at every step to output:
1. **Exact Component Attribution:** The signed percentage contribution of Throughput ($\Delta Q_{\text{tp}}$), Waiting Time ($\Delta Q_{\text{wait}}$), and Priority ($\Delta Q_{\text{prio}}$) to the winning margin $\Delta Q = \mathbf{w}^\top (\mathbf{Q}(s, a_1^*) - \mathbf{Q}(s, a_2^*))$.
2. **Instantaneous Profile-Flip Sensitivity:** Exact identification of which alternative profiles $\mathbf{w}_{m'} \neq \mathbf{w}_{\text{active}}$ would immediately flip the decision from $a_1^*$ to an alternative action.
3. **Physical State-Space Tipping Point:** By walking the discrete 1D state lattice along the opposing approach's clearance-time and waiting-time buckets and inverting $T_{\text{clear}}$, the narrator reports the exact number of additional trucks $\Delta N_{\text{truck}} = \left\lceil \frac{\Delta T_{\text{needed}} \cdot S_{\text{sat}}}{\text{PCU}_{\text{truck}}} \right\rceil$ or additional waiting seconds $\Delta W_{\text{needed}}$ required to flip the signal phase.

---

## 3. Research Questions & Experimental Hypotheses

* **$RQ_1$ (Seepage-Aware Green Allocation):** Does replacing static PCU/count state encoding with non-linear seepage-aware clearance time $T_{\text{clear}}$ reduce average delay and wasted green time in mixed two-wheeler/heavy-vehicle traffic?
* **$RQ_2$ (Zero-Retraining Profile Adaptation):** Can C-GPI over the 4-profile decomposed vector Q-bank switch objectives instantaneously at runtime (e.g., `commuter_priority` $\to$ `freight_priority`) and match or beat single-objective specialist agents trained from scratch?
* **$RQ_3$ (Hard Fairness Guarantee via Action Masking):** Does internal starvation action masking $\mathcal{M}(s,a)$ enforce $W_{95} \le W_{\max}$ with zero Bellman corruption compared to external heuristic overrides and soft reward penalties?
* **$RQ_4$ (Exact Counterfactual Auditability):** Can deterministic boundary introspection quantify both weight-space flip margins and physical vehicle tipping points in under $0.5\text{ ms}$ per decision?

---

## 4. Mandatory Simulation Rigor & Edge-Case Fixes

1. **Elimination of $T_c^2$ Double-Counting:** Physical intersection discharge inside `simulation.py` uses actual seepage/saturation discharge dynamics, never multiplying crossing time by priority weight squared.
2. **Dual-Stream Common Random Numbers (CRN):** `arrival_rng = random.Random(seed)` and `agent_rng = random.Random(seed + 10000)` are strictly isolated so all compared controllers face identical vehicle arrival streams.
3. **Uncensored Episode Waiting Time:** Delay metrics integrate waiting time across both discharged vehicles and residual queued vehicles at $t = T_{\max}$.