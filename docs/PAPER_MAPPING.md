# Paper-to-code mapping

This file is the main traceability document.

## Section 2 — 2-index stage-based formulation

Implementation: `src/fstsp/formulation/stage_based.py`.

| Paper item | Code concept | Notes |
|---|---|---|
| Parameters N, S, E, C, V, tau, tau', Dd, tL, tR | `FSTSPInstance` | Start/end depots are separate node IDs with same coordinate. |
| Stage horizon K=N+2 | `instance.stages` | 0-based in code; paper is conceptually 1-based. |
| X_i^k | `("X", k, i)` | Truck at location i in stage k. |
| x_ij | `("x", i, j)` | Truck arc. |
| phi_h | `("phi", h)` | Drone serves customer h. |
| A_h^i / B_h^i | `("A",h,i)` / `("B",h,i)` | Launch/recovery location. |
| Y_h^k / W_h^k | `("Y",h,k)` / `("W",h,k)` | Launch/recovery stage. |
| Z_kk' | `("Z",k,kp)` | Sortie spans stages k..k'. |
| a_i / d_i | `("a",i)` / `("d",i)` | Arrival/departure time. |
| Objective (1) | `c[d_E]=1` | Minimize completion time. |
| (2)-(9) | depot/routing/stage blocks | Truck route and connectivity. |
| (10)-(15) | drone consistency/endurance/non-overlap | One drone; no overlapping sorties. |
| (16)-(26) | drone-truck forcing | Couples stages, locations, assignments. |
| (27)-(33) | timing block | Big-M time synchronization. |
| (34)-(40) | strengthening block | Optional `strengthen=True`. |
| (41)-(55) | domains/bounds | Variable integrality and impossible-variable fixings. |

### Index conversion

Paper stage 1 corresponds to Python stage `0`; paper stage K corresponds to Python `K-1`. The algebra is preserved after this conversion.

## Section 3 — CMSA

Implementation: `src/fstsp/algorithms/cmsa/`.

| Paper step | Code |
|---|---|
| Construct (MTZ TSP + Stage-based MILP) | `construct.py::construct_solution`, `_integrate_drone_customers_stage_based` |
| Define components x, phi, A, B | `stage_based.py::solution_components` |
| Merge components | `age.py::AgeManager.mark_useful` |
| Fix inactive variables | `build_stage_model(... active_components=...)` |
| Solve restricted MIP | `solve_stage_model` |
| Adapt ages (Algorithm 1 Lines 11–18) | `age.py::AgeManager.adapt` (without protected set) |
| Track best solution | `algorithm.py::solve_cmsa` |

## Construction and reconstruction details

The paper specifies that the Construct step solves TSP on a sampled truck-customer subset using the MTZ formulation, and then integrates remaining customers using the 2-index stage-based formulation with a fixed number of stages equal to the TSP tour length plus two ($K = |C_{\text{truck}}| + 2$). This repo implements:

- Samples a fraction of truck customers (default 65%) with seeded RNG;
- Solves MTZ TSP for the truck tour on $\{S\} \cup C_{\text{truck}} \cup \{E\}$ using `exact_tsp_threshold = 60` so MTZ is always executed for all benchmark sizes ($n \le 50$);
- Integrates remaining drone customers via the exact 2-index stage-based MILP subproblem (`_integrate_drone_customers_stage_based`) with fixed truck route and $K = |C_{\text{truck}}| + 2$ stages, naturally discovering both consecutive and non-consecutive multi-stage sorties ($k \to k'$ with $k' > k$);
- Resolves unassigned drone customers via a resampling/promotion loop: promoting the customer with the largest detour to `truck_set` and re-solving MTZ TSP, strictly avoiding heuristic shortcuts;
- Tracks active search space through explicit `n_active_variables`, `n_active_constraints`, and `n_active_nonzeros` metrics.

## Experimental setup & solver transparency

1. **Solver differences**: The paper conducted experiments using commercial **IBM ILOG CPLEX 22.11** with `MIPEmphasis = 5` (Feasibility priority), 8 threads on an AMD Ryzen Threadripper PRO 5975WX with 252 GB RAM, and $t_{\text{MIP}} = 15\text{s}$. This reproduction utilizes **HiGHS** (via SciPy `milp`) with `mip_rel_gap = 0.02` in CMSA. This ensures an entirely open-source, reproducible implementation requiring no proprietary licenses, while preserving all 55 constraint equations.
2. **Benchmark instances (Table 3)**: The paper randomly generated 40 instances following Agatz et al.'s protocol ($n \in \{20, 30, 40, 50\}$, 10 instances each, uniform $[0, 100]^2$, depot $(50, 50)$, maxradius = 200%, truck speed 1, drone speed 2) but did not publish exact coordinate files or random seeds. Our 40 instances (generated via seeds 1–10) are an independent synthetic reproduction under the identical Agatz specification; objective values are statistical cross-instance comparisons rather than instance-by-instance matches.
3. **Model sizing (Table 4)**: Table 4 in the paper explicitly reports post-presolve dimensions from CPLEX 22.11 (`reported by CPLEX`), where fixed variables and redundant constraints are eliminated. In SciPy/HiGHS, we measure both raw static matrix dimensions and compact active subproblem dimensions (`n_active_variables`, `n_active_constraints`, `n_active_nonzeros`), confirming that `age_limit=2` strictly and monotonically produces a smaller search space than `age_limit=5`.
