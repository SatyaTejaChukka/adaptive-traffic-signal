"""
Pillar 2: Constrained Reward-Decomposed Vector Q-Learning with Multi-Profile Q-Bank,
Constrained Generalized Policy Improvement (C-GPI), and Hierarchical Starvation Masking.
"""

import pickle
import random
from typing import Dict, Tuple, List, Optional
import numpy as np

from rl_state import StateTuple, SeepageParams

# 6 Discrete Phase-Duration Macro-Actions: (Phase_Group_Str, Phase_Int, Green_Duration_Sec)
ACTION_SPACE: Dict[int, Tuple[str, int, int]] = {
    0: ("NS", 0, 5),
    1: ("NS", 0, 10),
    2: ("NS", 0, 15),
    3: ("EW", 1, 5),
    4: ("EW", 1, 10),
    5: ("EW", 1, 15),
}

# Canonical Multi-Objective Preference Weight Vectors w_m in Simplex^3
# Order of reward heads: [Throughput (r_tp), Waiting Penalty (r_wait), Class Priority (r_prio)]
CANONICAL_PROFILES: Dict[str, np.ndarray] = {
    "balanced":           np.array([0.40, 0.40, 0.20], dtype=np.float64),
    "commuter_priority":  np.array([0.55, 0.35, 0.10], dtype=np.float64),
    "freight_priority":   np.array([0.20, 0.25, 0.55], dtype=np.float64),
    "emergency_corridor": np.array([0.10, 0.15, 0.75], dtype=np.float64),
}


def compute_smdp_vector_reward(
    discharged_counts_by_class: Dict[str, int],
    cumulative_queue_pcu_sec: float,
    cumulative_wait_sec: float,
    cumulative_priority_wait_sec: float,
    elapsed_dt: float,
    phase_switched: bool,
    params: SeepageParams = SeepageParams(),
) -> np.ndarray:
    """
    Computes the 3-component duration-normalized SMDP reward rate vector:
      r_vec[0] = r_throughput (PCU discharge rate minus yellow switch loss)
      r_vec[1] = r_wait       (negative queue and waiting time pressure rate)
      r_vec[2] = r_priority   (priority-weighted discharge rate minus priority delay rate)
    """
    dt = max(float(elapsed_dt), 1.0)

    discharged_pcu = sum(
        count * params.pcu_weights.get(cls_name, 1.0)
        for cls_name, count in discharged_counts_by_class.items()
    )
    discharged_prio = sum(
        count * params.prio_weights.get(cls_name, 1.0) * params.pcu_weights.get(cls_name, 1.0)
        for cls_name, count in discharged_counts_by_class.items()
    )

    switch_overhead = 1.2 if phase_switched else 0.0

    # Head 0: Throughput efficiency rate
    r_tp = 3.5 * (discharged_pcu / dt) - switch_overhead

    # Head 1: Waiting time and queue control rate (negative penalty)
    mean_queue_pcu = cumulative_queue_pcu_sec / dt
    mean_wait = cumulative_wait_sec / dt
    r_wait = -(0.35 * mean_queue_pcu + 0.25 * mean_wait)

    # Head 2: Heterogeneous priority progression rate
    mean_prio_wait = cumulative_priority_wait_sec / dt
    r_prio = 3.0 * (discharged_prio / dt) - 0.30 * mean_prio_wait

    return np.array([r_tp, r_wait, r_prio], dtype=np.float64)


def compute_starvation_mask(
    wait_ns: float,
    wait_ew: float,
    t_clear_ns: float,
    t_clear_ew: float,
    w_max: float = 45.0,
    enable_mask: bool = True,
) -> np.ndarray:
    """
    Computes the Hierarchical Starvation Action Mask M(s,a) in {0.0, -inf}^6.
    Guarantees:
      1. If max(wait_ns, wait_ew) < w_max, all 6 actions are unmasked (0.0).
      2. If one or both approaches exceed w_max, selects the dominant starved phase
         d_starved = argmax(wait_ns, wait_ew) and masks out all opposing phase actions
         as well as durations shorter than required to flush d_starved.
      3. Never masks out all 6 actions (deadlock-free under dual starvation).
    """
    mask = np.zeros(6, dtype=np.float64)
    if not enable_mask or max(wait_ns, wait_ew) < w_max:
        return mask

    starved_dir = "NS" if wait_ns >= wait_ew else "EW"
    t_req = t_clear_ns if starved_dir == "NS" else t_clear_ew

    if t_req <= 5.0:
        min_green = 5
    elif t_req <= 10.0:
        min_green = 10
    else:
        min_green = 15

    for a_idx, (phase_str, _, green_sec) in ACTION_SPACE.items():
        if phase_str != starved_dir or green_sec < min_green:
            mask[a_idx] = -np.inf

    return mask


class DecomposedVectorQLearner:
    """
    Multi-Profile Reward-Decomposed Tabular Vector Q-Learner with Constrained
    Generalized Policy Improvement (C-GPI) for instantaneous zero-retraining
    traffic profile switching.
    """

    def __init__(
        self,
        active_profile: str = "balanced",
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon: float = 0.10,
        agent_seed: int = 10042,
    ) -> None:
        if active_profile not in CANONICAL_PROFILES:
            raise ValueError(f"Unknown profile: {active_profile}")
        self.active_profile: str = active_profile
        self.active_weights: np.ndarray = CANONICAL_PROFILES[active_profile].copy()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.rng: random.Random = random.Random(agent_seed)

        # Multi-Profile Q-Bank: profile_name -> {state_tuple -> np.ndarray of shape (6, 3)}
        self.q_bank: Dict[str, Dict[StateTuple, np.ndarray]] = {
            prof_name: {} for prof_name in CANONICAL_PROFILES
        }

    def _ensure_state(self, state: StateTuple) -> None:
        for prof_name in CANONICAL_PROFILES:
            if state not in self.q_bank[prof_name]:
                self.q_bank[prof_name][state] = np.zeros((6, 3), dtype=np.float64)

    def set_active_profile(
        self,
        profile_name: str,
        custom_weights: Optional[np.ndarray] = None,
    ) -> None:
        """
        Switches the active reward preference vector w_runtime in <1 ms with zero retraining.
        Supports both named canonical profiles and arbitrary continuous custom_weights in R^3.
        """
        if custom_weights is not None:
            w = np.asarray(custom_weights, dtype=np.float64)
            if w.shape != (3,) or np.sum(w) <= 0:
                raise ValueError("custom_weights must be a positive 3-element vector.")
            self.active_weights = w / np.sum(w)
            self.active_profile = profile_name
        else:
            if profile_name not in CANONICAL_PROFILES:
                raise ValueError(f"Unknown canonical profile: {profile_name}")
            self.active_profile = profile_name
            self.active_weights = CANONICAL_PROFILES[profile_name].copy()

    def compute_cgpi_q_values(
        self,
        state: StateTuple,
        weights: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates Constrained Generalized Policy Improvement (C-GPI) across the 4-profile
        Q-bank for the given preference vector w:
          q_cgpi[a] = max_{m in 1..M} (w^T Q_m(s, a))
        Returns:
          q_cgpi: shape (6,) scalarized C-GPI action values (before mask)
          effective_q_vectors: shape (6, 3) decomposed Q-vectors from the maximizing policy head
        """
        self._ensure_state(state)
        w = self.active_weights if weights is None else np.asarray(weights, dtype=np.float64)

        q_cgpi = np.full(6, -np.inf, dtype=np.float64)
        effective_q_vectors = np.zeros((6, 3), dtype=np.float64)

        for prof_name in CANONICAL_PROFILES:
            q_mat = self.q_bank[prof_name][state]  # shape (6, 3)
            scalar_vals = q_mat @ w                # shape (6,)
            improved = scalar_vals > q_cgpi
            q_cgpi[improved] = scalar_vals[improved]
            effective_q_vectors[improved] = q_mat[improved]

        return q_cgpi, effective_q_vectors

    def select_action(
        self,
        state: StateTuple,
        mask: np.ndarray,
        training: bool = False,
    ) -> Tuple[int, np.ndarray]:
        """
        Selects an action using C-GPI + Starvation Action Mask M(s,a).
        During exploration, only samples from unmasked valid actions {a | mask[a] == 0}.
        """
        valid_actions = [a for a in range(6) if np.isfinite(mask[a])]
        if not valid_actions:
            valid_actions = list(range(6))

        q_cgpi, effective_q_vectors = self.compute_cgpi_q_values(state)
        masked_q = q_cgpi + mask

        if training and self.rng.random() < self.epsilon:
            chosen_action = self.rng.choice(valid_actions)
            return int(chosen_action), effective_q_vectors

        chosen_action = int(np.argmax(masked_q))
        return chosen_action, effective_q_vectors

    def update(
        self,
        state: StateTuple,
        executed_action: int,
        reward_vec: np.ndarray,
        next_state: StateTuple,
        next_mask: np.ndarray,
        elapsed_dt: float,
    ) -> Dict[str, float]:
        """
        Performs synchronous off-policy SMDP Vector Bellman updates across all M=4
        profile heads in parallel using continuous-time discounting gamma^(dt / 5.0).
        """
        self._ensure_state(state)
        self._ensure_state(next_state)

        effective_gamma = self.gamma ** (max(float(elapsed_dt), 1.0) / 5.0)
        td_norms: Dict[str, float] = {}

        for prof_name, w_m in CANONICAL_PROFILES.items():
            q_curr = self.q_bank[prof_name][state][executed_action]  # shape (3,)
            q_next_mat = self.q_bank[prof_name][next_state]         # shape (6, 3)

            # Bootstrap action a'_m respects both profile weight w_m and starvation mask M(s', a')
            next_scalar = (q_next_mat @ w_m) + next_mask
            best_next_action = int(np.argmax(next_scalar))
            q_next_best = q_next_mat[best_next_action]              # shape (3,)

            td_target = reward_vec + effective_gamma * q_next_best
            td_error_vec = td_target - q_curr
            self.q_bank[prof_name][state][executed_action] = q_curr + self.alpha * td_error_vec
            td_norms[prof_name] = float(np.linalg.norm(td_error_vec))

        return td_norms

    def get_state_q_bank(self, state: StateTuple) -> Dict[str, np.ndarray]:
        """Returns a copy of all 4 profile (6, 3) Q-matrices for the given state."""
        self._ensure_state(state)
        return {prof: self.q_bank[prof][state].copy() for prof in CANONICAL_PROFILES}

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(self.q_bank, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            self.q_bank = pickle.load(f)