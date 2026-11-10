"""
Automated pytest suite verifying the mathematical and physical invariants of:
  - Pillar 1: Seepage-Aware Heterogeneous Queue Discharge (rl_state.py)
  - Pillar 2: Constrained Reward-Decomposed Vector Q-Learner (rl_agent.py)
  - Pillar 3: Deterministic Counterfactual Boundary Introspector (counterfactual_explainer.py)
"""

import numpy as np
import pytest

from rl_state import (
    SeepageParams,
    compute_seepage_clear_time,
    discretize_clear_time,
    build_seepage_rl_state,
)
from rl_agent import (
    CANONICAL_PROFILES,
    compute_smdp_vector_reward,
    compute_starvation_mask,
    DecomposedVectorQLearner,
)
from counterfactual_explainer import CounterfactualIntrospector


def test_vehicle_conservation_and_seepage_acceleration():
    """
    Verifies Catch 1 Fix:
      1. Two-wheelers never vanish when red_elapsed_sec > 0 (alpha_seep > 0).
      2. Allowing motorcycles to seep to the front during red strictly reduces T_clear
         compared to zero red time (alpha_seep == 0) because S_burst > S_sat / PCU_bike.
    """
    params = SeepageParams()
    counts = {"bike": 20, "rickshaw": 4, "car": 2, "bus": 0, "truck": 0}

    t_clear_zero_red, info_zero = compute_seepage_clear_time(counts, red_elapsed_sec=0.0, params=params)
    t_clear_with_seep, info_seep = compute_seepage_clear_time(counts, red_elapsed_sec=25.0, params=params)

    assert info_zero["alpha_seep"] == 0.0
    assert info_seep["alpha_seep"] > 0.50
    assert t_clear_with_seep > 3.0, "Two-wheelers must require positive physical clearance time"
    assert t_clear_with_seep < t_clear_zero_red, "Front seepage burst must reduce total clearance time"


def test_heavy_vehicle_pore_blockage_and_startup_lag():
    """
    Verifies that adding heavy trucks both increases startup inertia lag (t_lag)
    and suppresses two-wheeler seepage (alpha_seep) via pore-space blockage (rho_hv).
    """
    params = SeepageParams()
    light_queue = {"bike": 10, "rickshaw": 2, "car": 2, "bus": 0, "truck": 0}
    heavy_queue = {"bike": 10, "rickshaw": 2, "car": 2, "bus": 3, "truck": 5}

    t_light, info_light = compute_seepage_clear_time(light_queue, red_elapsed_sec=20.0, params=params)
    t_heavy, info_heavy = compute_seepage_clear_time(heavy_queue, red_elapsed_sec=20.0, params=params)

    assert info_light["t_lag"] == 0.0
    assert info_heavy["t_lag"] > 2.5
    assert info_heavy["alpha_seep"] < info_light["alpha_seep"]
    assert t_heavy > t_light + 10.0


def test_zero_retraining_profile_switch_flips_optimal_action():
    """
    Verifies Pillar 2 C-GPI:
      Switching runtime profile from 'commuter_priority' to 'freight_priority'
      immediately flips the optimal action in <1 ms without modifying any Q-values.
    """
    agent = DecomposedVectorQLearner(active_profile="commuter_priority")
    state = (2, 2, 0, 0, 1, 1)
    agent._ensure_state(state)

    # Action 1 (NS-10s) excels at high commuter throughput; Action 5 (EW-15s) excels at heavy freight priority
    q_matrix = np.array([
        [5.0, -3.0, 2.0],   # 0: NS-5s
        [9.2, -2.5, 2.5],   # 1: NS-10s (Commuter winner)
        [7.5, -3.2, 2.8],   # 2: NS-15s
        [4.0, -3.5, 5.5],   # 3: EW-5s
        [5.8, -2.8, 8.2],   # 4: EW-10s
        [6.2, -2.4, 9.8],   # 5: EW-15s (Freight winner)
    ], dtype=np.float64)

    for prof in CANONICAL_PROFILES:
        agent.q_bank[prof][state] = q_matrix.copy()

    mask = compute_starvation_mask(wait_ns=20.0, wait_ew=25.0, t_clear_ns=8.0, t_clear_ew=12.0, w_max=45.0)

    agent.set_active_profile("commuter_priority")
    a_commuter, _ = agent.select_action(state, mask, training=False)
    assert a_commuter == 1  # NS-10s

    # Instantaneous zero-retraining switch to freight_priority
    agent.set_active_profile("freight_priority")
    a_freight, _ = agent.select_action(state, mask, training=False)
    assert a_freight == 5  # EW-15s


def test_dual_starvation_mask_never_deadlocks():
    """
    Verifies Catch 3 Fix:
      When both NS and EW exceed W_max simultaneously, compute_starvation_mask
      selects the dominant starved phase and appropriate minimum green duration,
      leaving at least one valid action unmasked (0.0).
    """
    mask = compute_starvation_mask(
        wait_ns=52.0,
        wait_ew=58.0,
        t_clear_ns=11.0,
        t_clear_ew=9.2,
        w_max=45.0,
        enable_mask=True,
    )
    finite_actions = [a for a in range(6) if np.isfinite(mask[a])]
    assert len(finite_actions) >= 1, "Mask must never deadlock all 6 actions"
    # EW has higher wait (58s > 52s) and requires 9.2s clearance -> EW-10s (4) and EW-15s (5) valid
    assert finite_actions == [4, 5]


def test_counterfactual_attribution_and_physical_tipping_point():
    """
    Verifies Pillar 3 CounterfactualIntrospector:
      1. Component attribution percentages sum to 100% in magnitude.
      2. Detects alternative profiles that flip the decision.
      3. Computes positive integer truck tipping point and exact wait-to-mask delta.
    """
    agent = DecomposedVectorQLearner(active_profile="balanced")
    state = (1, 1, 0, 0, 1, 1)
    agent._ensure_state(state)

    q_matrix = np.array([
        [4.0, -2.0, 1.0],
        [8.5, -2.0, 3.0],   # Action 1: NS-10s (Chosen under balanced)
        [7.0, -2.5, 2.5],
        [3.5, -2.5, 4.0],
        [5.5, -2.2, 8.8],   # Action 4: EW-10s (Flips under freight_priority / emergency_corridor)
        [5.0, -2.8, 8.0],
    ], dtype=np.float64)
    for prof in CANONICAL_PROFILES:
        agent.q_bank[prof][state] = q_matrix.copy()

    mask = compute_starvation_mask(wait_ns=18.0, wait_ew=32.0, t_clear_ns=6.5, t_clear_ew=6.0, w_max=45.0)
    _, eff_q = agent.select_action(state, mask, training=False)

    introspector = CounterfactualIntrospector()
    report = introspector.explain_decision(
        state=state,
        q_vectors_active=eff_q,
        q_bank_state=agent.get_state_q_bank(state),
        mask=mask,
        active_profile="balanced",
        t_clear_ns=6.5,
        t_clear_ew=6.0,
        wait_ns=18.0,
        wait_ew=32.0,
        w_max=45.0,
    )

    assert report.chosen_action_idx == 1
    abs_pct_sum = sum(abs(v) for v in report.component_attribution_pct.values())
    assert pytest.approx(abs_pct_sum, abs=0.5) == 100.0
    flipping_names = [p[0] for p in report.profile_flips]
    assert "freight_priority" in flipping_names
    assert report.opposing_trucks_to_flip >= 1
    assert pytest.approx(report.opposing_wait_sec_to_mask_flip, abs=1e-3) == 13.0