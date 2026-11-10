"""
Comprehensive Benchmark & Ablation Experiment Runner for SeepageRL.

Evaluates 5 ablation models across 4 scenarios over 10 distinct seeds with Common
Random Numbers (CRN), recording full delay, throughput, fairness, and waste metrics to CSV.
"""

import os
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from config import (
    ACTION_SPACE,
    CANONICAL_PROFILES,
    MODEL_PATH,
    STARVATION_WAIT_THRESHOLD,
)
from rl_state import SeepageParams
from rl_agent import DecomposedVectorQLearner, compute_starvation_mask
from headless_runner import HeadlessTrafficEnv, EpisodeMetrics


class BaselineFixedTimerController:
    """Standard fixed-time cycle controller (alternating NS 10s, EW 10s)."""

    def __init__(self) -> None:
        self.next_action = 1  # NS-10s

    def select_action(self, state, telemetry) -> int:
        curr_phase = state[2]
        # Alternate: if was NS (0), next is EW-10s (4); if was EW (1), next is NS-10s (1)
        action = 4 if curr_phase == 0 else 1
        return action


class ScalarQLearnerBaseline:
    """Tabular scalar Q-learner baseline supporting M0, M1, M2 representations."""

    def __init__(self, model_variant: str = "M0_COUNT") -> None:
        self.model_variant = model_variant
        self.q_table: Dict[Any, np.ndarray] = {}

    def get_state_key(self, telemetry: Dict[str, Any], curr_phase: int, is_yellow: int):
        if self.model_variant == "M0_COUNT":
            # Raw vehicle counts
            ns_cnt = sum(telemetry["ns_counts"].values())
            ew_cnt = sum(telemetry["ew_counts"].values())
            return (min(ns_cnt // 5, 4), min(ew_cnt // 5, 4), curr_phase, is_yellow)
        elif self.model_variant == "M1_STATIC_PCU":
            # Linear static PCU
            pcu_w = {"bike": 0.4, "rickshaw": 0.8, "car": 1.0, "bus": 2.2, "truck": 2.5}
            ns_pcu = sum(cnt * pcu_w[k] for k, cnt in telemetry["ns_counts"].items())
            ew_pcu = sum(cnt * pcu_w[k] for k, cnt in telemetry["ew_counts"].items())
            return (min(int(ns_pcu // 5), 4), min(int(ew_pcu // 5), 4), curr_phase, is_yellow)
        else:  # M2_SEEPAGE_SCALAR
            b_ns = min(int(telemetry["t_clear_ns"] // 4.5), 3)
            b_ew = min(int(telemetry["t_clear_ew"] // 4.5), 3)
            return (b_ns, b_ew, curr_phase, is_yellow)

    def select_action(self, state, telemetry) -> int:
        wait_ns = telemetry["wait_ns"]
        wait_ew = telemetry["wait_ew"]

        # External heuristic starvation override
        if max(wait_ns, wait_ew) >= STARVATION_WAIT_THRESHOLD:
            # External override forces 10s green for starved direction
            return 1 if wait_ns >= wait_ew else 4

        key = self.get_state_key(telemetry, state[2], state[3])
        if key not in self.q_table:
            # Fallback heuristic: give green to approach with higher demand
            return 1 if telemetry["t_clear_ns"] >= telemetry["t_clear_ew"] else 4

        return int(np.argmax(self.q_table[key]))


def run_benchmark_suite(
    seeds: List[int] = list(range(1, 11)),
    scenarios: List[str] = ["S1_2W_SURGE", "S2_BALANCED", "S3_FREIGHT", "S4_REGIME_SHIFT"],
    output_dir: str = "experiments",
) -> pd.DataFrame:
    """
    Runs the full paired ablation benchmark across 5 models x 4 scenarios x 10 seeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = ["M0_COUNT", "M1_STATIC_PCU", "M2_SEEPAGE_SCALAR", "M3_VECTOR_UNMASKED", "M4_FULL_PROPOSED"]

    # Load pre-trained vector agent for M3 and M4
    vector_agent = DecomposedVectorQLearner(active_profile="balanced")
    if os.path.exists(MODEL_PATH):
        vector_agent.load(MODEL_PATH)
    else:
        print(f"Warning: {MODEL_PATH} not found, using untrained vector agent.")

    results: List[Dict[str, Any]] = []
    total_runs = len(models) * len(scenarios) * len(seeds)
    run_idx = 0
    t0 = time.time()

    print(f"\n=======================================================")
    print(f"Starting SeepageRL Benchmark: {total_runs} Paired Runs")
    print(f"=======================================================\n")

    for scen in scenarios:
        for model_id in models:
            for seed in seeds:
                run_idx += 1
                env = HeadlessTrafficEnv(sim_duration_sec=300.0, seed=seed, scenario_id=scen)
                state, telemetry = env.get_state_and_telemetry()
                done = False

                # Configure model controller
                if model_id in ("M0_COUNT", "M1_STATIC_PCU", "M2_SEEPAGE_SCALAR"):
                    controller = ScalarQLearnerBaseline(model_variant=model_id)
                else:
                    controller = vector_agent
                    # Under S4 regime shift, dynamically switch profile at t=150s
                    controller.set_active_profile("commuter_priority" if scen == "S4_REGIME_SHIFT" else "balanced")

                while not done:
                    # Under S4, trigger dynamic zero-retraining profile switch at t=150s
                    if scen == "S4_REGIME_SHIFT" and env.current_time >= 150.0 and model_id in ("M3_VECTOR_UNMASKED", "M4_FULL_PROPOSED"):
                        controller.set_active_profile("freight_priority")

                    if model_id in ("M0_COUNT", "M1_STATIC_PCU", "M2_SEEPAGE_SCALAR"):
                        action = controller.select_action(state, telemetry)
                    elif model_id == "M3_VECTOR_UNMASKED":
                        # Vector agent without hard starvation mask
                        zero_mask = np.zeros(6, dtype=np.float64)
                        action, _ = controller.select_action(state, zero_mask, training=False)
                    else:  # M4_FULL_PROPOSED
                        mask = compute_starvation_mask(
                            wait_ns=telemetry["wait_ns"],
                            wait_ew=telemetry["wait_ew"],
                            t_clear_ns=telemetry["t_clear_ns"],
                            t_clear_ew=telemetry["t_clear_ew"],
                            w_max=STARVATION_WAIT_THRESHOLD,
                            enable_mask=True,
                        )
                        action, _ = controller.select_action(state, mask, training=False)

                    next_state, _, done, next_telemetry = env.step(action)
                    state = next_state
                    telemetry = next_telemetry

                metrics = env.get_episode_metrics()
                results.append({
                    "model_id": model_id,
                    "scenario": scen,
                    "seed": seed,
                    "throughput": round(metrics.throughput_veh_per_sec, 4),
                    "mean_delay": round(metrics.mean_delay_sec, 2),
                    "w_95": round(metrics.w_95_sec, 2),
                    "w_max": round(metrics.w_max_sec, 2),
                    "wasted_green_ratio": round(metrics.wasted_green_ratio, 4),
                    "starvation_violations": metrics.starvation_violations,
                    "vehicles_discharged": metrics.total_vehicles_discharged,
                    "total_arrived": metrics.total_vehicles_arrived,
                })

            print(f"[{run_idx:3d}/{total_runs}] Completed {model_id:18s} on {scen:15s}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    total_time = time.time() - t0
    print(f"\nBenchmark finished in {total_time:.2f}s ({total_runs / total_time:.1f} runs/sec).")
    print(f"Results saved to {csv_path}\n")

    # Generate quick summary pivot table
    summary = df.groupby(["scenario", "model_id"]).agg({
        "mean_delay": ["mean", "std"],
        "throughput": ["mean", "std"],
        "w_95": ["mean", "std"],
        "starvation_violations": ["mean"],
    }).round(2)
    print("Benchmark Summary Table:")
    print(summary)

    return df


if __name__ == "__main__":
    run_benchmark_suite()

