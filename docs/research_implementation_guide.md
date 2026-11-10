# Research Implementation Guide (`README.md`)

This package contains the complete specification and ready-to-run Python implementation for the 3-pillar research paper upgrade of `SatyaTejaChukka/adaptive-traffic-signal`:

> **"Zero-Retraining Multi-Objective Traffic Signal Control for Heterogeneous Seepage Traffic via Reward-Decomposed Q-Learning and Counterfactual Action Masking"**

---

## 1. Included Files in This Workspace

### Planning & Architecture Specifications
1. **`prd.md`:** Governing equations for Pillar 1 ($T_{\text{clear}}$ Seepage + HV Lag), Pillar 2 (Multi-Profile Decomposed Vector Q-Learning + C-GPI + Starvation Mask $\mathcal{M}$), and Pillar 3 (Counterfactual Boundary Introspection), plus research questions $RQ_1\text{--}RQ_4$.
2. **`components.md`:** System block diagram, state-space cardinality ($576 \times 6 \times 4 \times 3$), memory footprint analysis ($331.8\text{ KB}$), and the 5-model ablation matrix (`M0_COUNT` through `M4_FULL_PROPOSED`).
3. **`functions.md`:** Exact type signatures, input/output contracts, and invariants for every module.
4. **`decisions.md`:** Architecture Decision Records (`ADR-001` through `ADR-005`) justifying the mathematical fixes.

### Drop-In Python Implementation Modules
5. **`rl_state.py`:** Complete implementation of Pillar 1 (`SeepageParams`, `compute_seepage_clear_time`, `discretize_clear_time`, `build_seepage_rl_state`, and `invert_clear_time_to_vehicles`).
6. **`rl_agent.py`:** Complete implementation of Pillar 2 (`DecomposedVectorQLearner`, `compute_smdp_vector_reward`, `compute_starvation_mask`, C-GPI runtime profile switching, and parallel SMDP vector Bellman updates).
7. **`counterfactual_explainer.py`:** Complete implementation of Pillar 3 (`CounterfactualIntrospector`, component attribution, profile-flip sensitivity, L2 hyperplane distance, and physical truck/wait tipping-point calculation).
8. **`tests/test_three_pillar_core.py`:** Comprehensive `pytest` test suite verifying every mathematical invariant across all 3 pillars.

---

## 2. Quick-Start Verification Commands

Copy the Python files into your repository root and run the automated verification test suite:

```bash
pytest tests/test_three_pillar_core.py -v
```

### Expected Test Suite Checks
* `test_vehicle_conservation_across_seepage_rates`: Verifies that two-wheelers never vanish as $\alpha_{\text{seep}}$ varies from $0.0$ to $0.75$ and that red-phase seepage strictly reduces clearance time $T_{\text{clear}}$.
* `test_heavy_vehicle_pore_blockage_and_startup_lag`: Verifies that adding heavy trucks increases startup inertia lag $\tau_{\text{lag}}$ and suppresses motorcycle front-seepage.
* `test_zero_retraining_profile_switch_flips_action`: Verifies that switching `agent.set_active_profile("freight_priority")` immediately flips the chosen signal phase in under $1\text{ ms}$ without retraining.
* `test_dual_starvation_mask_never_deadlocks`: Verifies that when both NS and EW exceed $W_{\max} = 45\text{ s}$ simultaneously, $\mathcal{M}(s,a)$ prioritizes the dominant starved approach with sufficient green duration to clear $T_{\text{clear}}$ and never masks out all 6 actions.
* `test_counterfactual_attribution_and_physical_tipping_point`: Verifies that component attributions sum to $100\%$ of the decision margin and that `opposing_trucks_to_flip` returns exact integer vehicle counts.