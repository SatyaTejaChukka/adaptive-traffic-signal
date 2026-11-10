"""
Pillar 3: Deterministic Counterfactual Boundary Introspection (EQPI).
Solves the exact linear hyperplane w^T (Q(s, a1*) - Q(s, a2*)) = 0 and inverts
the physical seepage clearance model T_clear to output exact vehicle/wait tipping points.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from rl_state import StateTuple, SeepageParams, invert_clear_time_to_vehicles
from rl_agent import ACTION_SPACE, CANONICAL_PROFILES


@dataclass(frozen=True)
class CounterfactualReport:
    chosen_action_idx: int
    chosen_phase: str
    chosen_green_sec: int
    runner_up_action_idx: int
    runner_up_phase: str
    runner_up_green_sec: int
    scalar_margin: float
    mask_forced_decision: bool
    component_attribution_pct: Dict[str, float]
    profile_flips: List[Tuple[str, str, int]]
    min_weight_shift_l2: float
    opposing_trucks_to_flip: int
    opposing_wait_sec_to_mask_flip: float
    narrative_summary: str


class CounterfactualIntrospector:
    """
    Computes exact linear attribution, runtime profile-flip sensitivity,
    and physical traffic tipping points in <0.5 ms per decision.
    """

    def __init__(self, params: SeepageParams = SeepageParams()) -> None:
        self.params = params

    def explain_decision(
        self,
        state: StateTuple,
        q_vectors_active: np.ndarray,          # shape (6, 3) from active C-GPI head
        q_bank_state: Dict[str, np.ndarray],   # profile_name -> shape (6, 3)
        mask: np.ndarray,                      # shape (6,)
        active_profile: str,
        t_clear_ns: float,
        t_clear_ew: float,
        wait_ns: float,
        wait_ew: float,
        w_max: float = 45.0,
    ) -> CounterfactualReport:
        w = CANONICAL_PROFILES[active_profile]
        unmasked_scalar_q = q_vectors_active @ w
        masked_scalar_q = unmasked_scalar_q + mask

        # Rank actions under the active constrained policy
        valid_indices = [a for a in range(6) if np.isfinite(masked_scalar_q[a])]
        ranked_valid = sorted(valid_indices, key=lambda a: float(masked_scalar_q[a]), reverse=True)
        a1 = ranked_valid[0]

        # Identify best runner-up action (preferring opposing phase for counterfactual contrast)
        chosen_phase_str, _, chosen_green = ACTION_SPACE[a1]
        all_ranked_unmasked = [int(a) for a in np.argsort(unmasked_scalar_q)[::-1]]
        opposing_candidates = [
            a for a in all_ranked_unmasked if ACTION_SPACE[a][0] != chosen_phase_str
        ]
        if len(ranked_valid) >= 2:
            a2 = ranked_valid[1]
        else:
            a2 = opposing_candidates[0]

        runner_up_phase_str, _, runner_up_green = ACTION_SPACE[a2]
        mask_forced = bool(np.any(~np.isfinite(mask)))

        # 1. Exact 3-Head Linear Attribution: w * (Q(s, a1) - Q(s, a2))
        delta_q_vec = q_vectors_active[a1] - q_vectors_active[a2]  # shape (3,)
        weighted_diff = w * delta_q_vec                            # shape (3,)
        scalar_margin = float(np.sum(weighted_diff))

        abs_total = float(np.sum(np.abs(weighted_diff))) + 1e-9
        attribution = {
            "throughput_pct": round(float(weighted_diff[0] / abs_total) * 100.0, 1),
            "wait_pct":       round(float(weighted_diff[1] / abs_total) * 100.0, 1),
            "priority_pct":   round(float(weighted_diff[2] / abs_total) * 100.0, 1),
        }

        # 2. Analytic L2 Distance to Decision Boundary Hyperplane w^T delta_q_vec = 0
        dq_norm_sq = float(np.dot(delta_q_vec, delta_q_vec))
        if dq_norm_sq > 1e-12 and scalar_margin > 0:
            min_weight_shift_l2 = float(abs(scalar_margin) / np.sqrt(dq_norm_sq))
        else:
            min_weight_shift_l2 = 0.0

        # 3. Instantaneous Profile-Flip Sensitivity across canonical profiles
        profile_flips: List[Tuple[str, str, int]] = []
        for prof_name, w_alt in CANONICAL_PROFILES.items():
            if prof_name == active_profile:
                continue
            # Evaluate C-GPI under alternative profile weight w_alt
            alt_cgpi = np.max(
                np.stack([q_mat @ w_alt for q_mat in q_bank_state.values()], axis=0),
                axis=0,
            )
            alt_best = int(np.argmax(alt_cgpi + mask))
            if alt_best != a1:
                alt_phase, _, alt_green = ACTION_SPACE[alt_best]
                profile_flips.append((prof_name, alt_phase, alt_green))

        # 4. Physical State-Space Tipping Point (Trucks & Waiting Time on Opposing Approach)
        opp_t_clear = t_clear_ew if chosen_phase_str == "NS" else t_clear_ns
        opp_wait = wait_ew if chosen_phase_str == "NS" else wait_ns

        cutoffs = self.params.clear_time_cutoffs
        next_cutoff = next((c for c in cutoffs if c > opp_t_clear), opp_t_clear + 4.5)
        trucks_to_flip = max(
            1, invert_clear_time_to_vehicles(opp_t_clear, next_cutoff, "truck", self.params)
        )
        wait_to_mask_flip = round(max(0.0, w_max - opp_wait), 1)

        # 5. Human-Readable Engineering Audit Narrative
        if mask_forced:
            narrative = (
                f"[STARVATION MASK ACTIVE] Selected {chosen_phase_str}-{chosen_green}s "
                f"because {chosen_phase_str} wait ({max(wait_ns, wait_ew):.1f}s) >= W_max ({w_max:.1f}s) "
                f"with T_clear={max(t_clear_ns, t_clear_ew):.1f}s."
            )
        else:
            flip_str = (
                f"Flips immediately to {profile_flips[0][1]}-{profile_flips[0][2]}s under '{profile_flips[0][0]}'."
                if profile_flips
                else "Decision is invariant across all 4 canonical profiles."
            )
            narrative = (
                f"Selected {chosen_phase_str}-{chosen_green}s over {runner_up_phase_str}-{runner_up_green}s "
                f"(Delta_Q={scalar_margin:+.2f} | Attribution: TP={attribution['throughput_pct']:+.1f}%, "
                f"Wait={attribution['wait_pct']:+.1f}%, Prio={attribution['priority_pct']:+.1f}%). "
                f"{flip_str} Opposing approach needs +{trucks_to_flip} truck(s) for next T_clear bucket "
                f"or +{wait_to_mask_flip:.1f}s wait to trigger hard starvation mask."
            )

        return CounterfactualReport(
            chosen_action_idx=a1,
            chosen_phase=chosen_phase_str,
            chosen_green_sec=chosen_green,
            runner_up_action_idx=a2,
            runner_up_phase=runner_up_phase_str,
            runner_up_green_sec=runner_up_green,
            scalar_margin=round(scalar_margin, 4),
            mask_forced_decision=mask_forced,
            component_attribution_pct=attribution,
            profile_flips=profile_flips,
            min_weight_shift_l2=round(min_weight_shift_l2, 4),
            opposing_trucks_to_flip=trucks_to_flip,
            opposing_wait_sec_to_mask_flip=wait_to_mask_flip,
            narrative_summary=narrative,
        )