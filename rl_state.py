"""
Pillar 1: Physics-Informed Seepage-Aware Heterogeneous Queue Discharge Model (T_clear)
and Action-Aligned Markov State Encoder for `adaptive-traffic-signal`.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple, Any

StateTuple = Tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class SeepageParams:
    """Physical and policy parameters for heterogeneous Indian traffic discharge."""
    pcu_weights: Dict[str, float] = field(default_factory=lambda: {
        "bike": 0.40,
        "rickshaw": 0.80,
        "car": 1.00,
        "bus": 2.20,
        "truck": 2.50,
    })
    prio_weights: Dict[str, float] = field(default_factory=lambda: {
        "bike": 1.00,
        "rickshaw": 1.10,
        "car": 1.00,
        "bus": 1.60,
        "truck": 1.80,
    })
    s_burst: float = 3.60          # Seeped 2W front-queue burst discharge rate (veh/sec)
    s_sat: float = 1.20            # Main platoon saturation flow rate (PCU/sec)
    alpha_max: float = 0.75        # Max fraction of 2Ws able to filter to the front
    tau_seep: float = 12.0         # Red-phase seepage accumulation time constant (sec)
    tau_lag_max: float = 3.50      # Max heavy-vehicle startup inertia lag (sec)
    lambda_hv: float = 0.45        # Exponential growth rate of HV startup lag
    beta_block: float = 1.50       # Lateral pore-blockage multiplier per HV share
    clear_time_cutoffs: Tuple[float, float, float] = (4.0, 8.5, 13.5)
    wait_time_cutoffs: Tuple[float, float] = (15.0, 40.0)


def compute_seepage_clear_time(
    vehicle_counts: Dict[str, int],
    red_elapsed_sec: float,
    params: SeepageParams = SeepageParams(),
) -> Tuple[float, Dict[str, float]]:
    """
    Computes the three-stage physical clearance time T_clear(d) in seconds:
      Stage 1: Front-seeped two-wheeler high-density burst clearing (t_burst)
      Stage 2: Heavy-vehicle startup inertia lag (t_lag)
      Stage 3: Non-seeped 2Ws + main platoon saturation flow (t_platoon)

    Strictly conserves all two-wheelers across all alpha_seep in [0, 1].
    """
    n_bike = max(0, int(vehicle_counts.get("bike", 0)))
    n_rick = max(0, int(vehicle_counts.get("rickshaw", 0)))
    n_car = max(0, int(vehicle_counts.get("car", 0)))
    n_bus = max(0, int(vehicle_counts.get("bus", 0)))
    n_truck = max(0, int(vehicle_counts.get("truck", 0)))

    n_hv = n_bus + n_truck
    n_total = n_bike + n_rick + n_car + n_hv
    if n_total == 0:
        return 0.0, {
            "alpha_seep": 0.0,
            "rho_hv": 0.0,
            "n_seep_front": 0.0,
            "t_burst": 0.0,
            "t_lag": 0.0,
            "t_platoon": 0.0,
            "platoon_pcu": 0.0,
        }

    # Heavy vehicles block lateral pore spaces between lanes during red phase
    rho_hv = min(1.0, (n_hv / float(n_total)) * params.beta_block)
    red_factor = 1.0 - math.exp(-max(0.0, float(red_elapsed_sec)) / params.tau_seep)
    alpha_seep = params.alpha_max * red_factor * (1.0 - rho_hv)

    # Rickshaws have half the lateral filtering permeability of motorcycles
    seeped_bikes = alpha_seep * n_bike
    seeped_ricks = 0.5 * alpha_seep * n_rick
    n_seep_front = seeped_bikes + seeped_ricks

    # Stage 1: High-density front-burst clearance time
    t_burst = n_seep_front / params.s_burst

    # Stage 2: Bounded non-linear heavy-vehicle startup lag
    t_lag = params.tau_lag_max * (1.0 - math.exp(-params.lambda_hv * n_hv))

    # Stage 3: Remaining non-seeped 2Ws + cars + buses + trucks at saturation flow
    rem_bikes = (1.0 - alpha_seep) * n_bike
    rem_ricks = (1.0 - 0.5 * alpha_seep) * n_rick
    platoon_pcu = (
        rem_bikes * params.pcu_weights["bike"]
        + rem_ricks * params.pcu_weights["rickshaw"]
        + n_car * params.pcu_weights["car"]
        + n_bus * params.pcu_weights["bus"]
        + n_truck * params.pcu_weights["truck"]
    )
    t_platoon = platoon_pcu / params.s_sat

    t_clear = float(t_burst + t_lag + t_platoon)
    breakdown = {
        "alpha_seep": round(alpha_seep, 4),
        "rho_hv": round(rho_hv, 4),
        "n_seep_front": round(n_seep_front, 3),
        "t_burst": round(t_burst, 3),
        "t_lag": round(t_lag, 3),
        "t_platoon": round(t_platoon, 3),
        "platoon_pcu": round(platoon_pcu, 3),
    }
    return t_clear, breakdown


def discretize_clear_time(
    t_clear_sec: float,
    params: SeepageParams = SeepageParams(),
) -> int:
    """
    Maps continuous clearance time T_clear(d) into discrete bucket {0, 1, 2, 3}
    aligned with the {5s, 10s, 15s} green duration action space.
    """
    c0, c1, c2 = params.clear_time_cutoffs
    if t_clear_sec <= c0:
        return 0
    elif t_clear_sec <= c1:
        return 1
    elif t_clear_sec <= c2:
        return 2
    return 3


def discretize_wait_time(
    max_wait_sec: float,
    params: SeepageParams = SeepageParams(),
) -> int:
    """Maps maximum directional waiting time into discrete bucket {0, 1, 2}."""
    w0, w1 = params.wait_time_cutoffs
    if max_wait_sec <= w0:
        return 0
    elif max_wait_sec <= w1:
        return 1
    return 2


def aggregate_class_counts(vehicles: List[Any]) -> Dict[str, int]:
    """Counts waiting vehicles by class from a list of Vehicle objects or dicts."""
    counts = {"bike": 0, "rickshaw": 0, "car": 0, "bus": 0, "truck": 0}
    for v in vehicles:
        v_cls = v.get("vehicle_class") if isinstance(v, dict) else getattr(v, "vehicle_class", "car")
        if v_cls in counts:
            counts[v_cls] += 1
    return counts


def extract_max_wait(vehicles: List[Any]) -> float:
    """Extracts maximum waiting time (in seconds) from a list of Vehicle objects or dicts."""
    if not vehicles:
        return 0.0
    return float(
        max(
            (v.get("wait_time", 0.0) if isinstance(v, dict) else getattr(v, "wait_time", 0.0))
            for v in vehicles
        )
    )


def build_seepage_rl_state(
    queues: Dict[str, List[Any]],
    red_elapsed: Dict[str, float],
    current_phase: int,
    is_yellow: int,
    params: SeepageParams = SeepageParams(),
) -> Tuple[StateTuple, Dict[str, Any]]:
    """
    Constructs the 6-tuple Markov state s_t = (b_T_NS, b_T_EW, phase, yellow, b_W_NS, b_W_EW)
    and returns physical telemetry for Pillar 2 masking and Pillar 3 counterfactuals.
    """
    ns_vehicles = queues.get("North", []) + queues.get("South", [])
    ew_vehicles = queues.get("East", []) + queues.get("West", [])

    ns_counts = aggregate_class_counts(ns_vehicles)
    ew_counts = aggregate_class_counts(ew_vehicles)

    red_ns = float(red_elapsed.get("NS", 0.0))
    red_ew = float(red_elapsed.get("EW", 0.0))

    t_clear_ns, info_ns = compute_seepage_clear_time(ns_counts, red_ns, params)
    t_clear_ew, info_ew = compute_seepage_clear_time(ew_counts, red_ew, params)

    b_t_ns = discretize_clear_time(t_clear_ns, params)
    b_t_ew = discretize_clear_time(t_clear_ew, params)

    wait_ns = extract_max_wait(ns_vehicles)
    wait_ew = extract_max_wait(ew_vehicles)

    b_w_ns = discretize_wait_time(wait_ns, params)
    b_w_ew = discretize_wait_time(wait_ew, params)

    state: StateTuple = (
        b_t_ns,
        b_t_ew,
        int(current_phase),
        int(is_yellow),
        b_w_ns,
        b_w_ew,
    )
    telemetry = {
        "t_clear_ns": t_clear_ns,
        "t_clear_ew": t_clear_ew,
        "wait_ns": wait_ns,
        "wait_ew": wait_ew,
        "ns_counts": ns_counts,
        "ew_counts": ew_counts,
        "ns_breakdown": info_ns,
        "ew_breakdown": info_ew,
    }
    return state, telemetry


def invert_clear_time_to_vehicles(
    current_t_clear: float,
    target_t_clear: float,
    vehicle_class: str = "truck",
    params: SeepageParams = SeepageParams(),
) -> int:
    """
    Computes the exact integer number of additional vehicles of `vehicle_class`
    needed to push `current_t_clear` above `target_t_clear`.
    """
    delta_t = max(0.0, target_t_clear - current_t_clear)
    if delta_t <= 1e-9:
        return 0
    pcu_k = params.pcu_weights.get(vehicle_class, 1.0)
    # Account for both platoon saturation time and marginal startup lag if heavy vehicle
    marginal_sec_per_veh = pcu_k / params.s_sat
    return int(math.ceil(delta_t / max(marginal_sec_per_veh, 1e-6)))