"""
High-Performance Headless Traffic Intersection Simulation & Benchmark Engine for SeepageRL.

Enables ultra-fast, 100% reproducible execution of traffic scenarios with Common Random
Numbers (CRN), physical 3-stage queue discharge, and zero Pygame overhead.
"""

from dataclasses import dataclass, field
import math
import os
import random
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from config import (
    ACTION_SPACE,
    CANONICAL_PROFILES,
    CLEAR_TIME_CUTOFFS,
    DEFAULT_GREEN_DURATION,
    ENABLE_STARVATION_MASK,
    GAMMA,
    GREEN_DURATION_OPTIONS,
    MODEL_PATH,
    PCU_WEIGHTS,
    PRIORITY_WEIGHTS,
    STARVATION_WAIT_THRESHOLD,
    WAIT_TIME_CUTOFFS,
    YELLOW_DURATION,
)
from rl_state import (
    SeepageParams,
    StateTuple,
    build_seepage_rl_state,
    compute_seepage_clear_time,
    discretize_clear_time,
    discretize_wait_time,
)
from rl_agent import (
    DecomposedVectorQLearner,
    compute_smdp_vector_reward,
    compute_starvation_mask,
)
from counterfactual_explainer import CounterfactualIntrospector


# Scenario traffic profiles
SCENARIO_CONFIGS = {
    "S1_2W_SURGE": {
        "description": "High two-wheeler surge (morning peak commuter mix)",
        "arrival_prob_ns": 0.45,
        "arrival_prob_ew": 0.35,
        "class_probs": {
            "bike": 0.55,
            "rickshaw": 0.15,
            "car": 0.20,
            "bus": 0.05,
            "truck": 0.05,
        },
    },
    "S2_BALANCED": {
        "description": "Balanced urban mixed traffic",
        "arrival_prob_ns": 0.38,
        "arrival_prob_ew": 0.38,
        "class_probs": {
            "bike": 0.35,
            "rickshaw": 0.10,
            "car": 0.35,
            "bus": 0.10,
            "truck": 0.10,
        },
    },
    "S3_FREIGHT": {
        "description": "Heavy industrial freight corridor",
        "arrival_prob_ns": 0.30,
        "arrival_prob_ew": 0.45,
        "class_probs": {
            "bike": 0.15,
            "rickshaw": 0.05,
            "car": 0.30,
            "bus": 0.25,
            "truck": 0.25,
        },
    },
    "S4_REGIME_SHIFT": {
        "description": "Dynamic regime shift: S1 (0-150s) -> S3 (150-300s)",
        "shift_time_sec": 150.0,
    },
}


@dataclass
class QueuedVehicle:
    vehicle_class: str
    arrival_time: float
    wait_time: float = 0.0


@dataclass
class EpisodeMetrics:
    total_vehicles_arrived: int = 0
    total_vehicles_discharged: int = 0
    total_delay_sec: float = 0.0
    mean_delay_sec: float = 0.0
    w_95_sec: float = 0.0
    w_max_sec: float = 0.0
    throughput_veh_per_sec: float = 0.0
    wasted_green_sec: float = 0.0
    wasted_green_ratio: float = 0.0
    starvation_violations: int = 0
    decisions_count: int = 0
    mask_activations: int = 0
    class_delay: Dict[str, float] = field(default_factory=dict)
    class_discharged: Dict[str, int] = field(default_factory=dict)


class HeadlessTrafficEnv:
    """
    Microscopic discrete-event simulation environment for a 4-way intersection.
    Decoupled from Pygame for deterministic, ultra-fast parallel execution.
    """

    def __init__(
        self,
        sim_duration_sec: float = 300.0,
        yellow_duration: float = 3.0,
        seed: int = 42,
        scenario_id: str = "S2_BALANCED",
        params: SeepageParams = SeepageParams(),
    ) -> None:
        self.sim_duration_sec = sim_duration_sec
        self.yellow_duration = yellow_duration
        self.seed = seed
        self.scenario_id = scenario_id
        self.params = params

        # Dual-Stream Common Random Numbers
        self.arrival_rng = random.Random(seed)
        self.agent_rng = random.Random(seed + 10000)

        self.reset()

    def reset(self) -> None:
        self.current_time = 0.0
        self.current_phase = 0  # 0: NS, 1: EW
        self.is_yellow = 0
        self.red_elapsed = {"NS": 0.0, "EW": 0.0}

        # Queues per approach: 'North', 'South', 'East', 'West'
        self.queues: Dict[str, List[QueuedVehicle]] = {
            "North": [],
            "South": [],
            "East": [],
            "West": [],
        }

        # Telemetry logs
        self.completed_vehicle_delays: List[float] = []
        self.class_delays: Dict[str, List[float]] = {k: [] for k in self.params.pcu_weights}
        self.class_discharged_counts: Dict[str, int] = {k: 0 for k in self.params.pcu_weights}
        self.total_arrived = 0
        self.wasted_green_total = 0.0
        self.total_green_time = 0.0
        self.starvation_events = 0
        self.mask_activations = 0
        self.decisions_count = 0

    def _sample_scenario_config(self) -> Dict[str, Any]:
        if self.scenario_id == "S4_REGIME_SHIFT":
            shift_t = SCENARIO_CONFIGS["S4_REGIME_SHIFT"]["shift_time_sec"]
            return SCENARIO_CONFIGS["S1_2W_SURGE"] if self.current_time < shift_t else SCENARIO_CONFIGS["S3_FREIGHT"]
        return SCENARIO_CONFIGS.get(self.scenario_id, SCENARIO_CONFIGS["S2_BALANCED"])

    def _spawn_arrivals(self, dt_sec: float) -> None:
        cfg = self._sample_scenario_config()
        prob_ns = cfg["arrival_prob_ns"]
        prob_ew = cfg["arrival_prob_ew"]
        class_probs = cfg["class_probs"]
        classes = list(class_probs.keys())
        weights = list(class_probs.values())

        # Each second, check arrival on each approach
        num_seconds = int(math.ceil(dt_sec))
        for _ in range(num_seconds):
            # NS approaches (North, South)
            for app in ("North", "South"):
                if self.arrival_rng.random() < prob_ns:
                    v_cls = self.arrival_rng.choices(classes, weights=weights, k=1)[0]
                    self.queues[app].append(QueuedVehicle(vehicle_class=v_cls, arrival_time=self.current_time))
                    self.total_arrived += 1

            # EW approaches (East, West)
            for app in ("East", "West"):
                if self.arrival_rng.random() < prob_ew:
                    v_cls = self.arrival_rng.choices(classes, weights=weights, k=1)[0]
                    self.queues[app].append(QueuedVehicle(vehicle_class=v_cls, arrival_time=self.current_time))
                    self.total_arrived += 1

    def _accumulate_waiting_times(self, dt_sec: float) -> Tuple[float, float, float]:
        """
        Increments wait times for all currently queued vehicles across dt_sec.
        Returns: (cum_queue_pcu_sec, cum_wait_sec, cum_priority_wait_sec)
        """
        cum_queue_pcu = 0.0
        cum_wait = 0.0
        cum_prio_wait = 0.0

        for app, v_list in self.queues.items():
            for v in v_list:
                v.wait_time += dt_sec
                pcu = self.params.pcu_weights.get(v.vehicle_class, 1.0)
                prio = self.params.prio_weights.get(v.vehicle_class, 1.0)

                cum_queue_pcu += pcu * dt_sec
                cum_wait += v.wait_time * dt_sec
                cum_prio_wait += prio * v.wait_time * dt_sec

        return cum_queue_pcu, cum_wait, cum_prio_wait

    def get_state_and_telemetry(self) -> Tuple[StateTuple, Dict[str, Any]]:
        return build_seepage_rl_state(
            queues=self.queues,
            red_elapsed=self.red_elapsed,
            current_phase=self.current_phase,
            is_yellow=self.is_yellow,
            params=self.params,
        )

    def step(self, action_idx: int) -> Tuple[StateTuple, np.ndarray, bool, Dict[str, Any]]:
        """
        Executes a macro phase-duration action.
        Returns: (next_state, reward_vector, done, telemetry_info)
        """
        chosen_phase_str, target_phase_int, green_duration = ACTION_SPACE[action_idx]
        self.decisions_count += 1

        phase_switched = (target_phase_int != self.current_phase)
        yellow_time = self.yellow_duration if phase_switched else 0.0
        macro_dt = float(green_duration) + yellow_time
        self.total_green_time += float(green_duration)

        # 1. Handle Yellow Clearance Transition if phase changes
        if phase_switched:
            self.is_yellow = 1
            self._spawn_arrivals(yellow_time)
            self._accumulate_waiting_times(yellow_time)
            # Advance red timers
            if self.current_phase == 0:  # was NS
                self.red_elapsed["NS"] += yellow_time
            else:
                self.red_elapsed["EW"] += yellow_time
            self.current_time += yellow_time
            self.current_phase = target_phase_int
            self.is_yellow = 0

        # Reset red elapsed for the newly active green phase
        active_dir_str = "NS" if self.current_phase == 0 else "EW"
        opposing_dir_str = "EW" if active_dir_str == "NS" else "NS"
        self.red_elapsed[active_dir_str] = 0.0

        # 2. Spawn arrivals and accumulate waiting time during green duration
        self._spawn_arrivals(green_duration)
        cum_q_pcu, cum_wait, cum_prio_wait = self._accumulate_waiting_times(green_duration)
        self.red_elapsed[opposing_dir_str] += float(green_duration)
        self.current_time += float(green_duration)

        # Check starvation violation threshold
        wait_ns = max(
            [0.0] + [v.wait_time for v in self.queues["North"] + self.queues["South"]]
        )
        wait_ew = max(
            [0.0] + [v.wait_time for v in self.queues["East"] + self.queues["West"]]
        )
        if max(wait_ns, wait_ew) > STARVATION_WAIT_THRESHOLD:
            self.starvation_events += 1

        # 3. Physical 3-Stage Queue Discharge from Green Approach
        active_apps = ("North", "South") if self.current_phase == 0 else ("East", "West")
        discharged_counts = {k: 0 for k in self.params.pcu_weights}
        green_budget_sec = float(green_duration)

        for app in active_apps:
            queue = self.queues[app]
            if not queue:
                continue

            app_counts = {k: sum(1 for v in queue if v.vehicle_class == k) for k in self.params.pcu_weights}
            t_clear_app, breakdown = compute_seepage_clear_time(
                app_counts,
                red_elapsed_sec=self.red_elapsed[active_dir_str],
                params=self.params,
            )

            # Stage 1: Front seepage burst
            n_seep = breakdown["n_seep_front"]
            t_burst = breakdown["t_burst"]
            burst_budget = min(green_budget_sec, t_burst)

            seep_discharged = int(math.floor(burst_budget * self.params.s_burst))
            rem_green = max(0.0, green_budget_sec - t_burst - breakdown["t_lag"])
            platoon_pcu_budget = rem_green * self.params.s_sat

            # Discharge front-filtered two-wheelers first
            seeped_cleared = 0
            new_queue: List[QueuedVehicle] = []
            for v in queue:
                if v.vehicle_class in ("bike", "rickshaw") and seeped_cleared < seep_discharged:
                    seeped_cleared += 1
                    discharged_counts[v.vehicle_class] += 1
                    self.class_discharged_counts[v.vehicle_class] += 1
                    self.completed_vehicle_delays.append(v.wait_time)
                    self.class_delays[v.vehicle_class].append(v.wait_time)
                else:
                    new_queue.append(v)

            # Stage 3: Discharge main platoon vehicles within remaining PCU budget
            cleared_queue: List[QueuedVehicle] = []
            used_pcu = 0.0
            for v in new_queue:
                pcu = self.params.pcu_weights.get(v.vehicle_class, 1.0)
                if used_pcu + pcu <= platoon_pcu_budget:
                    used_pcu += pcu
                    discharged_counts[v.vehicle_class] += 1
                    self.class_discharged_counts[v.vehicle_class] += 1
                    self.completed_vehicle_delays.append(v.wait_time)
                    self.class_delays[v.vehicle_class].append(v.wait_time)
                else:
                    cleared_queue.append(v)

            self.queues[app] = cleared_queue

            # Track wasted green if clearance time was significantly shorter than green duration
            if t_clear_app < green_budget_sec:
                wasted = green_budget_sec - t_clear_app
                self.wasted_green_total += wasted

        # 4. Compute SMDP Vector Reward
        reward_vec = compute_smdp_vector_reward(
            discharged_counts_by_class=discharged_counts,
            cumulative_queue_pcu_sec=cum_q_pcu,
            cumulative_wait_sec=cum_wait,
            cumulative_priority_wait_sec=cum_prio_wait,
            elapsed_dt=macro_dt,
            phase_switched=phase_switched,
            params=self.params,
        )

        done = (self.current_time >= self.sim_duration_sec)
        next_state, telemetry = self.get_state_and_telemetry()

        return next_state, reward_vec, done, telemetry

    def get_episode_metrics(self) -> EpisodeMetrics:
        """
        Computes uncensored metrics including residual vehicles still in queue at T_max.
        """
        all_delays = list(self.completed_vehicle_delays)
        for app, v_list in self.queues.items():
            for v in v_list:
                all_delays.append(v.wait_time)
                self.class_delays[v.vehicle_class].append(v.wait_time)

        total_discharged = sum(self.class_discharged_counts.values())
        mean_delay = float(np.mean(all_delays)) if all_delays else 0.0
        w_95 = float(np.percentile(all_delays, 95)) if all_delays else 0.0
        w_max = float(np.max(all_delays)) if all_delays else 0.0
        tp = float(total_discharged) / max(self.current_time, 1.0)
        wasted_ratio = (
            float(self.wasted_green_total) / max(self.total_green_time, 1.0)
        )

        class_mean_delay = {
            cls_name: float(np.mean(delays)) if delays else 0.0
            for cls_name, delays in self.class_delays.items()
        }

        return EpisodeMetrics(
            total_vehicles_arrived=self.total_arrived,
            total_vehicles_discharged=total_discharged,
            total_delay_sec=float(np.sum(all_delays)) if all_delays else 0.0,
            mean_delay_sec=mean_delay,
            w_95_sec=w_95,
            w_max_sec=w_max,
            throughput_veh_per_sec=tp,
            wasted_green_sec=self.wasted_green_total,
            wasted_green_ratio=wasted_ratio,
            starvation_violations=self.starvation_events,
            decisions_count=self.decisions_count,
            mask_activations=self.mask_activations,
            class_delay=class_mean_delay,
            class_discharged=dict(self.class_discharged_counts),
        )


def train_seepage_vector_agent(
    num_episodes: int = 500,
    save_path: str = MODEL_PATH,
    verbose: bool = True,
) -> DecomposedVectorQLearner:
    """
    Trains the 4-profile DecomposedVectorQLearner across diverse scenarios using
    parallel SMDP vector Bellman updates and saves the Q-bank.
    """
    agent = DecomposedVectorQLearner(active_profile="balanced", alpha=0.15, gamma=GAMMA, epsilon=0.15)
    scenarios = ["S1_2W_SURGE", "S2_BALANCED", "S3_FREIGHT"]

    start_time = time.time()
    for ep in range(1, num_episodes + 1):
        scen = scenarios[ep % len(scenarios)]
        # Cycle through profiles to guarantee balanced state-space coverage
        prof_name = list(CANONICAL_PROFILES.keys())[ep % len(CANONICAL_PROFILES)]
        agent.set_active_profile(prof_name)

        env = HeadlessTrafficEnv(sim_duration_sec=300.0, seed=ep, scenario_id=scen)
        state, telemetry = env.get_state_and_telemetry()
        done = False

        while not done:
            mask = compute_starvation_mask(
                wait_ns=telemetry["wait_ns"],
                wait_ew=telemetry["wait_ew"],
                t_clear_ns=telemetry["t_clear_ns"],
                t_clear_ew=telemetry["t_clear_ew"],
                w_max=STARVATION_WAIT_THRESHOLD,
                enable_mask=ENABLE_STARVATION_MASK,
            )
            action, _ = agent.select_action(state, mask, training=True)
            next_state, reward_vec, done, next_telemetry = env.step(action)

            next_mask = compute_starvation_mask(
                wait_ns=next_telemetry["wait_ns"],
                wait_ew=next_telemetry["wait_ew"],
                t_clear_ns=next_telemetry["t_clear_ns"],
                t_clear_ew=next_telemetry["t_clear_ew"],
                w_max=STARVATION_WAIT_THRESHOLD,
                enable_mask=ENABLE_STARVATION_MASK,
            )

            # Parallel SMDP vector update across all 4 profile heads
            elapsed_dt = float(ACTION_SPACE[action][2]) + (3.0 if state[2] != next_state[2] else 0.0)
            agent.update(state, action, reward_vec, next_state, next_mask, elapsed_dt)

            state = next_state
            telemetry = next_telemetry

        # Decay exploration rate epsilon
        if agent.epsilon > 0.02:
            agent.epsilon *= 0.995

        if verbose and (ep % 100 == 0 or ep == num_episodes):
            elapsed = time.time() - start_time
            print(f"Episode {ep:4d}/{num_episodes} | Epsilon: {agent.epsilon:.3f} | States visited: {len(agent.q_bank['balanced'])} | Time: {elapsed:.1f}s")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    if verbose:
        print(f"Successfully saved 4-profile vector Q-bank to {save_path}")
    return agent


if __name__ == "__main__":
    print("Testing Headless Simulation Engine & Training Seepage Vector Agent...")
    train_seepage_vector_agent(num_episodes=200, verbose=True)

