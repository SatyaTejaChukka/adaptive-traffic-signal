# Function Contracts, Signatures, and Pseudocode (`functions.md`)

**Project Repository:** `SatyaTejaChukka/adaptive-traffic-signal`  
**Language Standard:** Python 3.10+ (`numpy`, `dataclasses`, `typing`)  

---

## 1. Pillar 1 Contracts: Seepage-Aware Queue Discharge (`rl_state.py`)

### 1.1 `SeepageParams` & `ClearanceBreakdown`
```python
@dataclass(frozen=True)
class SeepageParams:
    pcu_weights: Dict[str, float]      # {"bike": 0.4, "rickshaw": 0.8, "car": 1.0, "bus": 2.2, "truck": 2.5}
    prio_weights: Dict[str, float]     # {"bike": 1.0, "rickshaw": 1.1, "car": 1.0, "bus": 1.6, "truck": 1.8}
    s_burst: float = 2.4               # Front-seeped 2W burst discharge rate (veh/sec)
    s_sat: float = 1.2                 # Main platoon saturation flow rate (PCU/sec)
    alpha_max: float = 0.75            # Maximum fraction of 2Ws that can seep to front
    tau_seep: float = 12.0             # Red-time seepage saturation time constant (sec)
    tau_lag_max: float = 3.5           # Maximum heavy-vehicle startup inertia lag (sec)
    lambda_hv: float = 0.45            # Heavy-vehicle lag growth rate
    beta_block: float = 1.5            # Pore-space blockage multiplier per HV fraction
    clear_time_cutoffs: Tuple[float, float, float] = (4.0, 8.5, 13.5)
    wait_time_cutoffs: Tuple[float, float] = (15.0, 40.0)
```

### 1.2 `compute_seepage_clear_time`
* **Signature:**
  ```python
  def compute_seepage_clear_time(
      vehicle_counts: Dict[str, int],
      red_elapsed_sec: float,
      params: SeepageParams = SeepageParams()
  ) -> Tuple[float, Dict[str, float]]:
  ```
* **Mathematical Guarantee:**
  1. When `sum(vehicle_counts.values()) == 0`, returns `0.0`.
  2. Total two-wheelers $N_{\text{bike}} + N_{\text{rickshaw}}$ are strictly conserved between `n_seep_front` and `platoon_pcu` for all $\alpha_{\text{seep}} \in [0, 1]$.
  3. Increasing `red_elapsed_sec` (when $\rho_{\text{HV}} \lt 1$) increases $\alpha_{\text{seep}}$ and strictly decreases $T_{\text{clear}}$ because $S_{\text{burst}} \gt \frac{S_{\text{sat}}}{\text{PCU}_{\text{bike}}}$.

---

## 2. Pillar 2 Contracts: Constrained Decomposed Vector Q-Learner (`rl_agent.py`)

### 2.1 `compute_smdp_vector_reward`
* **Signature:**
  ```python
  def compute_smdp_vector_reward(
      discharged_counts_by_class: Dict[str, int],
      cumulative_queue_pcu_sec: float,
      cumulative_wait_sec: float,
      cumulative_priority_wait_sec: float,
      elapsed_dt: float,
      phase_switched: bool,
      params: SeepageParams = SeepageParams()
  ) -> np.ndarray:
  ```
* **Postcondition:** Returns a 3-element `np.ndarray` of shape `(3,)` with `dtype=np.float64`:
  * `r_vec[0]` ($r_{\text{tp}}$): Duration-normalized PCU discharge rate minus yellow switch overhead.
  * `r_vec[1]` ($r_{\text{wait}}$): Negative duration-normalized waiting and queue pressure rate.
  * `r_vec[2]` ($r_{\text{prio}}$): Priority-weighted discharge bonus minus priority-weighted queue delay rate.

### 2.2 `compute_starvation_mask`
* **Signature:**
  ```python
  def compute_starvation_mask(
      wait_ns: float,
      wait_ew: float,
      t_clear_ns: float,
      t_clear_ew: float,
      w_max: float = 45.0,
      enable_mask: bool = True
  ) -> np.ndarray:
  ```
* **Postcondition:** Returns `mask` of shape `(6,)` with values in $\{0.0, -\infty\}$. Always leaves at least one valid action ($0.0$) unmasked, preventing $-\infty$ deadlock even when both $\text{wait}_{\text{NS}} \ge W_{\max}$ and $\text{wait}_{\text{EW}} \ge W_{\max}$.

### 2.3 `DecomposedVectorQLearner.select_action` & `update`
* **Signatures:**
  ```python
  class DecomposedVectorQLearner:
      def select_action(
          self,
          state: StateTuple,
          mask: np.ndarray,
          training: bool = False
      ) -> Tuple[int, np.ndarray]:
          ...
      def update(
          self,
          state: StateTuple,
          executed_action: int,
          reward_vec: np.ndarray,
          next_state: StateTuple,
          next_mask: np.ndarray,
          elapsed_dt: float
      ) -> Dict[str, float]:
          ...
  ```
* **Invariants:**
  * `select_action` evaluates C-GPI across all $M=4$ profile Q-heads:
    $$Q_{\text{CGPI}}(s,a) = \max_{m \in \{1,\dots,4\}} \mathbf{w}_{\text{active}}^\top \mathbf{Q}_m(s,a)$$
    and adds `mask[a]` prior to $\arg\max_a$. Even during $\epsilon$-greedy exploration (`training=True`), random action sampling is restricted strictly to unmasked actions $\{a \mid \mathcal{M}(s,a) = 0\}$.
  * `update` performs off-policy vector Bellman updates across all 4 profile heads simultaneously using `executed_action` and SMDP discount factor $\gamma^{\Delta t / 5.0}$.

---

## 3. Pillar 3 Contracts: Counterfactual Boundary Introspector (`counterfactual_explainer.py`)

### 3.1 `CounterfactualReport` & `explain_decision`
* **Signature:**
  ```python
  def explain_decision(
      state: StateTuple,
      q_vectors_active: np.ndarray,          # shape (6, 3)
      q_bank_state: Dict[str, np.ndarray],   # profile_name -> shape (6, 3)
      mask: np.ndarray,                      # shape (6,)
      active_profile: str,
      t_clear_ns: float,
      t_clear_ew: float,
      wait_ns: float,
      wait_ew: float,
      w_max: float = 45.0,
      params: SeepageParams = SeepageParams()
  ) -> CounterfactualReport:
  ```
* **Outputs in `CounterfactualReport`:**
  * `chosen_action_idx: int`, `chosen_phase: str`, `chosen_green_sec: int`
  * `runner_up_action_idx: int`, `runner_up_phase: str`, `runner_up_green_sec: int`
  * `scalar_margin: float` ($\Delta Q = \mathbf{w}^\top (\mathbf{Q}(s, a_1^*) - \mathbf{Q}(s, a_2^*))$)
  * `component_attribution_pct: Dict[str, float]` (`throughput_pct`, `wait_pct`, `priority_pct`)
  * `profile_flips: List[Tuple[str, str, int]]` (which alternative profiles immediately flip $a_1^*$)
  * `min_weight_shift_l2: float` (analytic minimum L2 perturbation $\|\Delta \mathbf{w}\|_2$ on the hyperplane $\Delta \mathbf{Q}^\top \mathbf{w} = 0$ required to tie $a_1^*$ and $a_2^*$)
  * `opposing_trucks_to_flip: int` (exact additional heavy trucks on opposing approach required to shift $T_{\text{clear}}$ bucket or flip action)
  * `opposing_wait_sec_to_mask_flip: float` (exact additional seconds of waiting time until $\mathcal{M}(s,a)$ forces a phase switch)
  * `narrative_summary: str` (human-readable engineering audit string for the live dashboard)