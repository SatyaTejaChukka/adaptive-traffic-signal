# Adaptive Traffic Signal Timer with RL

This project simulates an adaptive traffic signal controller using Reinforcement Learning.

The final version focuses on:

- traffic generation inside the simulator
- lane and phase control at a four-way junction
- Q-learning based signal phase and green-timer selection
- safe yellow transitions between phase changes
- a fixed-timer fallback mode for baseline comparison

## Project Goal

The goal is to improve intersection throughput compared with a fixed or rule-based timer by using RL to decide:

- which phase should receive green next
- how long the green signal should remain active

## Current Architecture

The project is organized around these files inside `Code/RL/traffic_rl`:

- `simulation.py`
  Runs the traffic simulation, vehicle generation, UI, and signal control loop.
- `config.py`
  Stores training, evaluation, timing, and reward settings.
- `rl_agent.py`
  Implements the tabular Q-learning agent.
- `rl_state.py`
  Encodes queue and phase information into a compact RL state.
- `reward.py`
  Computes congestion-based reward values.

## Running the Simulation

Move into the simulator folder:

```powershell
cd "Code\RL\traffic_rl"
```

Install the runtime dependency:

```powershell
pip install -r requirements.txt
```

Run the simulator:

```powershell
python simulation.py
```

## Final Evaluation Mode

The final submission/demo configuration uses:

- `USE_RL = True`
- `RL_TRAINING_MODE = False`
- `RL_EVAL_MODE = True`
- `LOAD_MODEL_ON_START = True`
- `SAVE_MODEL_ON_EXIT = False`
- `EPSILON_START = 0.0`

The learned Q-table is loaded from:

- `Code/RL/traffic_rl/models/traffic_qtable.pkl`

## Reported Comparison

Example comparison after training:

- RL controller: `306` vehicles passed in `300` seconds, throughput `1.02`
- Fixed timer: `298` vehicles passed in `300` seconds, throughput `0.9933`

This indicates a small throughput improvement for the RL controller in the simulation.
