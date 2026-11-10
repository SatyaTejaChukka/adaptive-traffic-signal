"""
Central configuration for SeepageRL: Zero-Retraining Multi-Objective Traffic Signal Control.
"""

import os
import numpy as np

# Operational Mode
USE_RL = True
RL_TRAINING_MODE = False
RL_EVAL_MODE = True
LOAD_MODEL_ON_START = True
SAVE_MODEL_ON_EXIT = False

# RL Hyperparameters
ALPHA = 0.15
GAMMA = 0.95
EPSILON_START = 0.05
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Signal Timing Constraints
YELLOW_DURATION = 3
DEFAULT_GREEN_DURATION = 10
MIN_GREEN_DURATION = 5
MAX_GREEN_DURATION = 15
GREEN_DURATION_OPTIONS = (5, 10, 15)

# Hard Starvation Constraint Bound (Pillar 2 Action Masking)
STARVATION_WAIT_THRESHOLD = 45.0  # W_max in seconds
ENABLE_STARVATION_MASK = True

# Canonical Multi-Objective Preference Profiles w_m in Simplex^3
# Order of reward heads: [Throughput (r_tp), Waiting Penalty (r_wait), Class Priority (r_prio)]
CANONICAL_PROFILES = {
    "balanced": np.array([0.40, 0.40, 0.20], dtype=np.float64),
    "commuter_priority": np.array([0.55, 0.35, 0.10], dtype=np.float64),
    "freight_priority": np.array([0.20, 0.25, 0.55], dtype=np.float64),
    "emergency_corridor": np.array([0.10, 0.15, 0.75], dtype=np.float64),
}

# Active profile name (can be switched instantly at runtime via hotkeys 1-4)
TRAFFIC_PROFILE = "balanced"

# Profile Descriptions for Dashboard UI
PROFILE_METADATA = {
    "balanced": {
        "title": "Balanced Mixed Flow",
        "desc": "Equal fairness between throughput and queue delay across all modes.",
        "weights": (0.40, 0.40, 0.20),
        "color": (46, 204, 113),  # Green
    },
    "commuter_priority": {
        "title": "Peak Commuter Rush",
        "desc": "Prioritizes passenger cars and two-wheelers for rapid clearance.",
        "weights": (0.55, 0.35, 0.10),
        "color": (52, 152, 219),  # Blue
    },
    "freight_priority": {
        "title": "Industrial Logistics Corridor",
        "desc": "Heavily weights heavy trucks and buses to reduce stopping inertia.",
        "weights": (0.20, 0.25, 0.55),
        "color": (230, 126, 34),  # Orange
    },
    "emergency_corridor": {
        "title": "Emergency Preemption Mode",
        "desc": "Maximum priority weighting for rapid corridor clearance.",
        "weights": (0.10, 0.15, 0.75),
        "color": (231, 76, 60),   # Red
    },
}

# Physical Heterogeneous Vehicle Characteristics
PCU_WEIGHTS = {
    "bike": 0.40,
    "rickshaw": 0.80,
    "car": 1.00,
    "bus": 2.20,
    "truck": 2.50,
}

PRIORITY_WEIGHTS = {
    "bike": 1.00,
    "rickshaw": 1.10,
    "car": 1.00,
    "bus": 1.60,
    "truck": 1.80,
}

# Pillar 1 Seepage Dynamics Parameters
S_BURST = 3.60          # Seeped 2W front-queue burst discharge rate (veh/sec)
S_SAT = 1.20            # Main platoon saturation flow rate (PCU/sec)
ALPHA_MAX = 0.75        # Max fraction of 2Ws able to filter to the front
TAU_SEEP = 12.0         # Red-phase seepage accumulation time constant (sec)
TAU_LAG_MAX = 3.50      # Max heavy-vehicle startup inertia lag (sec)
LAMBDA_HV = 0.45        # Exponential growth rate of HV startup lag
BETA_BLOCK = 1.50       # Lateral pore-blockage multiplier per HV share

CLEAR_TIME_CUTOFFS = (4.0, 8.5, 13.5)
WAIT_TIME_CUTOFFS = (15.0, 40.0)

# Action Space (6 Macro-Actions)
ACTION_SPACE = {
    0: ("NS", 0, 5),
    1: ("NS", 0, 10),
    2: ("NS", 0, 15),
    3: ("EW", 1, 5),
    4: ("EW", 1, 10),
    5: ("EW", 1, 15),
}

# Storage paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "seepage_vector_qbank.pkl")
LEGACY_MODEL_PATH = os.path.join(MODEL_DIR, "traffic_qtable.pkl")
