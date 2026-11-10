"""
Reward computation module for SeepageRL.
Routes duration-normalized SMDP vector rewards across Throughput, Waiting Penalty,
and Class Priority heads, with scalarization support for legacy and ablation baselines.
"""

from typing import Dict, Optional
import numpy as np

from rl_state import SeepageParams
from rl_agent import compute_smdp_vector_reward, CANONICAL_PROFILES


def compute_vector_reward(
    discharged_counts_by_class: Dict[str, int],
    cumulative_queue_pcu_sec: float,
    cumulative_wait_sec: float,
    cumulative_priority_wait_sec: float,
    elapsed_dt: float,
    phase_switched: bool,
    params: SeepageParams = SeepageParams(),
) -> np.ndarray:
    """Computes the 3-component duration-normalized SMDP reward vector [r_tp, r_wait, r_prio]."""
    return compute_smdp_vector_reward(
        discharged_counts_by_class=discharged_counts_by_class,
        cumulative_queue_pcu_sec=cumulative_queue_pcu_sec,
        cumulative_wait_sec=cumulative_wait_sec,
        cumulative_priority_wait_sec=cumulative_priority_wait_sec,
        elapsed_dt=elapsed_dt,
        phase_switched=phase_switched,
        params=params,
    )


def scalarize_vector_reward(reward_vec: np.ndarray, profile_name: str = "balanced") -> float:
    """Scalarizes a 3-component reward vector using canonical profile weights."""
    weights = CANONICAL_PROFILES.get(profile_name, CANONICAL_PROFILES["balanced"])
    return float(np.dot(reward_vec, weights))


# Backward compatibility shim for legacy tests or baselines
def compute_reward(queue_counts, waiting_times, phase_changed, previous_queue_counts=None, illegal_state=False):
    total_queue = float(sum(queue_counts.values()))
    total_wait = float(sum(waiting_times.values()))
    r = -(total_queue + 0.1 * total_wait)
    if phase_changed:
        r -= 1.0
    if illegal_state:
        r -= 50.0
    return r
