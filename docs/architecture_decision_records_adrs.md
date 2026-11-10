# Architecture Decision Records (`decisions.md`)

**Project Repository:** `SatyaTejaChukka/adaptive-traffic-signal`  
**Version:** 2.0.0 (3-Pillar Architecture)  

---

## ADR-001: Unify Repository into a Cohesive 3-Pillar Architecture

* **Status:** Accepted (Supersedes v1.0 Single-Index Plan)
* **Context:** Reviewing 2023–2026 traffic control literature showed that individual claims around basic class weights, meta-RL profile switching, fairness penalties, or attention heatmaps face heavy prior art. However, combining (1) Physics-Informed Two-Wheeler Seepage + HV Lag Discharge ($T_{\text{clear}}$), (2) Constrained Multi-Profile Vector Q-Learning with C-GPI and Starvation Masking, and (3) Exact Linear Counterfactual Tipping-Point Introspection addresses three open gaps simultaneously.
* **Decision:** Upgrade the repository to the 3-pillar architecture while keeping everything tabular, deterministic, and executable on a standard CPU in pure Python/NumPy.

---

## ADR-002: Enforce Vehicle Conservation in Two-Wheeler Seepage Equation

* **Status:** Accepted
* **Context:** An initial draft of $T_{\text{clear}}$ wrote $\frac{N_{\text{2W}}(1-\alpha_{\text{seep}})}{S_{\text{burst}}} + \frac{\sum_{k \notin \text{2W}} N_k \text{PCU}_k}{S_{\text{sat}}}$, which caused two-wheelers to vanish as $\alpha_{\text{seep}} \to 1$.
* **Decision:** Partition two-wheelers into the front-seeped cohort $\alpha_{\text{seep}}(N_{\text{bike}} + 0.5 N_{\text{rickshaw}})$ discharging at $S_{\text{burst}} = 2.4\text{ veh/s}$ and the non-seeped tail cohort discharging within the main platoon at $S_{\text{sat}} = 1.2\text{ PCU/s}$, modulated by heavy-vehicle pore-blockage $\rho_{\text{HV}}$.
* **Consequences:** Guarantees strict conservation of vehicles for all $\alpha_{\text{seep}} \in [0, 1]$ and reflects empirical Indian mixed-traffic queuing physics.

---

## ADR-003: Multi-Profile Vector Q-Bank with Constrained Generalized Policy Improvement (C-GPI)

* **Status:** Accepted
* **Context:** A single 3-vector $\mathbf{Q}(s,a)$ bootstrapped under a single weight vector $\mathbf{w}_{\text{train}}$ only learns the value of future states under $\pi_{\mathbf{w}_{\text{train}}}$, and taking component-wise $\max_{a'} Q_k(s',a')$ violates the single-action constraint of MDPs.
* **Decision:** Maintain $M = 4$ vector Q-tables $\mathbf{Q}_m(s,a) \in \mathbb{R}^3$ (one per canonical profile $\mathbf{w}_m$), update all 4 tables in parallel on every transition using off-policy vector Bellman backups, and apply C-GPI ($a^* = \arg\max_a [\max_m \mathbf{w}_{\text{runtime}}^\top \mathbf{Q}_m(s,a) + \mathcal{M}(s,a)]$) at runtime.
* **Consequences:** Zero-retraining switching across both canonical profiles and arbitrary continuous weight vectors $\mathbf{w}_{\text{runtime}} \in \Delta^2$ is mathematically exact and takes under $0.2\text{ ms}$.

---

## ADR-004: Hierarchical Starvation Action Mask $\mathcal{M}(s,a)$ Inside the Bellman Operator

* **Status:** Accepted
* **Context:** External starvation overrides bypass the RL agent, causing the agent to neither anticipate starvation nor select an appropriate green duration when overridden. Moreover, if both NS and EW exceed $W_{\max}$, a naive mask blocks all actions.
* **Decision:** Embed $\mathcal{M}(s,a) \in \{0, -\infty\}^6$ directly inside both action selection and the Bellman bootstrap $\arg\max_{a'}$. Resolve dual-starvation ties hierarchically via $d_{\text{starved}} = \arg\max_d W_d$ and mask out durations shorter than $g_{\min}(T_{\text{clear}}(d_{\text{starved}}))$.
* **Consequences:** Guarantees at least one valid action exists in every state and eliminates off-policy Bellman corruption.

---

## ADR-005: Analytic Hyperplane + 1D State-Lattice Inversion for Counterfactuals

* **Status:** Accepted
* **Context:** Traffic engineers need both policy-level explanations ("Why did Priority beat Throughput?") and physical sensor-level tipping points ("How many more trucks on EW would trigger green?").
* **Decision:** Combine closed-form hyperplane projection ($\Delta \mathbf{w}_{\min} = -\frac{\mathbf{w}^\top \Delta \mathbf{Q}}{\|\Delta \mathbf{Q}\|_2^2} \Delta \mathbf{Q}$) for weight-space sensitivity with 1D discrete state-bucket stepping and $T_{\text{clear}}$ inversion ($\Delta N_{\text{truck}} = \lceil \Delta T_{\text{needed}} S_{\text{sat}} / \text{PCU}_{\text{truck}} \rceil$) for physical traffic tipping points.