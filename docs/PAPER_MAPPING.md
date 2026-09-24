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
| Construct | `construct.py::construct_solution` |
| Define components x, phi, A, B | `stage_based.py::solution_components` |
| Merge components | `age.py::AgeManager.mark_useful` |
| Fix inactive variables | `build_stage_model(... active_components=...)` |
| Solve restricted MIP | `solve_stage_model` |
| Adapt ages | `AgeManager.adapt` |
| Track best solution | `algorithm.py::solve_cmsa` |

## Known reconstruction choices

The paper says the Construct step solves TSP on a sampled truck-customer subset with an MTZ formulation, but does not publish the sampling policy and integration code. This repo:

- samples a deterministic fraction under a seeded RNG;
- uses MTZ MILP for small subsets;
- falls back to Nearest Neighbor + 2-opt for larger subsets;
- assigns at most one drone customer to each consecutive truck edge during construction;
- lets the restricted stage-MILP perform the exact optimization under the active component pool.

These are explicitly reimplementation choices, not claims about unpublished author code.
