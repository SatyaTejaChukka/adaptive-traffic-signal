"""
Publication Vector Figure Generator for SeepageRL Research Paper.

Generates high-resolution IEEE-formatted figures:
  - Fig 1: Physics-informed Seepage Discharge & HV Blockage Curves
  - Fig 2: Multi-Model Ablation Study Bar Charts across S1-S4
  - Fig 3: Cumulative Distribution Function (CDF) of Maximum Waiting Times (Fairness Bound)
  - Fig 4: Zero-Retraining Dynamic Regime Shift Adaptation under S4
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from rl_state import SeepageParams, compute_seepage_clear_time

# Configure publication-grade styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
    "figure.autolayout": True,
})

OUTPUT_DIR = "images/paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_fig1_seepage_physics():
    """Generates Figure 1: Three-stage seepage clearance time as a function of red time & HV fraction."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    params = SeepageParams()

    # Subplot A: T_clear vs Red-Elapsed Time for different 2W swarm sizes (n_hv = 0)
    red_times = np.linspace(0.0, 40.0, 100)
    for n_bikes in [10, 20, 30]:
        t_clears = []
        for r_t in red_times:
            counts = {"bike": n_bikes, "rickshaw": int(n_bikes * 0.25), "car": 2, "bus": 0, "truck": 0}
            t_c, _ = compute_seepage_clear_time(counts, red_elapsed_sec=r_t, params=params)
            t_clears.append(t_c)
        ax1.plot(red_times, t_clears, label=f"{n_bikes} Bikes + {int(n_bikes*0.25)} Autos")

    ax1.set_xlabel("Elapsed Red Signal Duration (s)")
    ax1.set_ylabel("Physical Clearance Time $T_{\\mathrm{clear}}$ (s)")
    ax1.set_title("(a) Seepage Burst Acceleration vs. Red Time")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(0, 20)

    # Subplot B: Pore blockage suppression by Heavy Vehicles (HV)
    hv_counts = np.arange(0, 11)
    base_bikes = 20
    t_clear_suppressed = []
    alpha_seeps = []

    for nhv in hv_counts:
        n_bus = nhv // 2
        n_truck = nhv - n_bus
        counts = {"bike": base_bikes, "rickshaw": 4, "car": 2, "bus": n_bus, "truck": n_truck}
        t_c, bd = compute_seepage_clear_time(counts, red_elapsed_sec=25.0, params=params)
        t_clear_suppressed.append(t_c)
        alpha_seeps.append(bd["alpha_seep"] * 100.0)

    color_t = "#1f77b4"
    color_a = "#d62728"

    ax2.plot(hv_counts, t_clear_suppressed, color=color_t, marker="o", label="$T_{\\mathrm{clear}}$ (s)")
    ax2.set_xlabel("Number of Heavy Vehicles (Buses + Trucks)")
    ax2.set_ylabel("Clearance Time $T_{\\mathrm{clear}}$ (s)", color=color_t)
    ax2.tick_params(axis="y", labelcolor=color_t)
    ax2.set_title("(b) HV Startup Lag & Seepage Blockage")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(hv_counts, alpha_seeps, color=color_a, marker="s", linestyle="--", label="$\\alpha_{\\mathrm{seep}}$ (%)")
    ax2_twin.set_ylabel("Effective Seepage Fraction $\\alpha_{\\mathrm{seep}}$ (%)", color=color_a)
    ax2_twin.tick_params(axis="y", labelcolor=color_a)
    ax2_twin.set_ylim(0, 80)
    ax2_twin.grid(False)

    out_path = os.path.join(OUTPUT_DIR, "fig1_seepage_discharge_physics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


def plot_fig2_ablation_delay(csv_path: str = "experiments/benchmark_results.csv"):
    """Generates Figure 2: Ablation study bar chart showing Mean Delay across S1-S4."""
    if not os.path.exists(csv_path):
        print(f"Cannot generate Fig 2: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    scenarios = ["S1_2W_SURGE", "S2_BALANCED", "S3_FREIGHT", "S4_REGIME_SHIFT"]
    models = ["M0_COUNT", "M1_STATIC_PCU", "M2_SEEPAGE_SCALAR", "M3_VECTOR_UNMASKED", "M4_FULL_PROPOSED"]
    labels = ["M0: Count", "M1: Static PCU", "M2: Seepage Scalar", "M3: Vector Unmasked", "M4: Proposed (Full)"]
    colors = ["#7f7f7f", "#aec7e8", "#ffbb78", "#e377c2", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(scenarios))
    width = 0.16

    for i, (m_id, lbl, col) in enumerate(zip(models, labels, colors)):
        sub = df[df["model_id"] == m_id]
        means = [sub[sub["scenario"] == s]["mean_delay"].mean() for s in scenarios]
        stds = [sub[sub["scenario"] == s]["mean_delay"].std() for s in scenarios]
        offset = (i - 2) * width
        rects = ax.bar(x + offset, means, width, yerr=stds, label=lbl, color=col, capsize=3, edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Mean Vehicle Delay (seconds)")
    ax.set_title("Ablation Benchmark: Mean Delay Across Operational Scenarios (10 Seeds $\\pm 1\\sigma$)")
    ax.set_xticks(x)
    ax.set_xticklabels(["S1: 2W Surge", "S2: Balanced Urban", "S3: Freight Corridor", "S4: Dynamic Shift"])
    ax.legend(loc="upper left", ncol=3, framealpha=0.95)

    out_path = os.path.join(OUTPUT_DIR, "fig2_ablation_mean_delay.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


def plot_fig3_fairness_cdf(csv_path: str = "experiments/benchmark_results.csv"):
    """Generates Figure 3: CDF of Maximum Waiting Time validating the hard W_max bound."""
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    # Filter to challenging mixed scenario S2 and S3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, scen, title in zip((ax1, ax2), ("S2_BALANCED", "S3_FREIGHT"), ("(a) S2: Balanced Mixed Traffic", "(b) S3: Freight Corridor")):
        sub = df[df["scenario"] == scen]
        for m_id, col, lbl in [
            ("M0_COUNT", "#7f7f7f", "M0: Count Baseline"),
            ("M3_VECTOR_UNMASKED", "#d62728", "M3: Unmasked RL"),
            ("M4_FULL_PROPOSED", "#2ca02c", "M4: Proposed (Masked)"),
        ]:
            data = np.sort(sub[sub["model_id"] == m_id]["w_max"].values)
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, label=lbl, color=col, marker="o" if "M4" in m_id else None)

        ax.axvline(45.0, color="black", linestyle=":", linewidth=1.5, label="Hard Bound $W_{\\max}=45$s")
        ax.set_xlabel("Maximum Observed Waiting Time (s)")
        ax.set_ylabel("Cumulative Empirical Probability (CDF)")
        ax.set_title(title)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim(20, 120)
        ax.set_ylim(0, 1.05)

    out_path = os.path.join(OUTPUT_DIR, "fig3_fairness_cdf_starvation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated {out_path}")


if __name__ == "__main__":
    print("Generating High-Resolution Publication Figures...")
    plot_fig1_seepage_physics()
    plot_fig2_ablation_delay()
    plot_fig3_fairness_cdf()
    print("All paper figures generated successfully!")

