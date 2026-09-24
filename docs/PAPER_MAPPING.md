# Paper-to-code mapping

This file is the main traceability document mapping the published paper directly to the implementation.

---

## Section 2 — 2-index stage-based formulation

Implementation: [`src/fstsp/formulation/stage_based.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/formulation/stage_based.py).

| Paper item | Code concept | Implementation Notes |
|---|---|---|
| Parameters $N, S, E, C, V, \tau, \tau', D_d, t_L, t_R$ | `FSTSPInstance` | Start/end depots are separate node IDs with identical coordinates. |
| Stage horizon $K = N + 2$ | `instance.stages` | 0-based in code ($0 \dots K-1$); paper is conceptually 1-based ($1 \dots K$). |
| $X_i^k$ | `("X", k, i)` | Binary: truck visits node $i$ at stage $k$. |
| $x_{ij}$ | `("x", i, j)` | Binary: truck travels directly from node $i$ to $j$. |
| $\phi_h$ | `("phi", h)` | Binary: drone serves customer $h$. |
| $A_h^i / B_h^i$ | `("A", h, i)` / `("B", h, i)` | Binary: drone launch / recovery node for customer $h$. |
| $Y_h^k / W_h^k$ | `("Y", h, k)` / `("W", h, k)` | Binary: drone launch / recovery stage for customer $h$. |
| $Z_{kk'}$ | `("Z", k, kp)` | Continuous: drone sortie spans stages $k \dots k'$ ($k < k'$). |
| $a_i / d_i$ | `("a", i)` / `("d", i)` | Continuous: arrival / departure time at node $i$. |
| Objective (1) | `c[d_E] = 1.0` | Minimize total trip makespan ($d_E$). |
| (2)–(9) | Depot/routing/stages | Truck route starts at $S$, ends at $E$, flow and stage connectivity. |
| (10)–(15) | Drone consistency/endurance/non-overlap | Exactly 1 drone; non-overlapping sorties across stages. |
| (16)–(26) | Drone-truck forcing | Couples stage, node, and vehicle assignments. |
| (27)–(33) | Timing block | Big-M physical synchronization of truck and drone. |
| (34)–(40) | Strengthening block | Valid inequalities, including Equations (34) and (35). |
| (41)–(55) | Domains/bounds | Variable integrality and impossible-variable fixings to zero. |

### Equation (35) Implementation Proof

Equation (35) states:
$$\sum_{k=1}^K k X_E^k + \sum_{h \in C} \phi_h = N + 2$$
The constant $N + 2$ represents the total number of original customers plus start and end depot. If the truck arrives at $E$ at 1-based stage $k_E$, it serves $k_E - 2$ customers by truck. Since all $N$ customers are partitioned between truck and drone ($k_E - 2 + \sum \phi_h = N$), the sum $k_E + \sum \phi_h$ is identically $N + 2$. The code strictly uses `instance.n + 2` as the constant RHS for all subproblems and models.

---

## Section 3 — CMSA Architecture

Implementation: [`src/fstsp/algorithms/cmsa/`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/algorithms/cmsa/).

| Paper step | Code | Implementation Details |
|---|---|---|
| Construct (MTZ TSP + Stage-based MILP) | `construct.py::construct_solution`, `_integrate_drone_customers_stage_based` | Solves MTZ TSP on sampled subset; integrates drone customers via stage-based MILP with $K = \|C_{\text{truck}}\| + 2$. |
| Define components $x, \phi, A, B$ | `stage_based.py::solution_components` | Extracts truck arcs, drone customers, launch and recovery nodes. |
| Merge components | `age.py::AgeManager.mark_useful` | Sets age $= 0$ for components in constructed and MIP solutions. |
| Fix inactive variables | `build_stage_model(active_components=...)` | Variables with age $< 0$ fixed to zero ($UB = 0$). |
| Solve restricted MIP | `solve_stage_model` | Solves restricted stage model over full stage horizon $K = N + 2$. |
| Adapt ages (Algorithm 1 Lines 11–18) | `age.py::AgeManager.adapt` | Increments all active components; disables components reaching `age_limit`. |
| Track best solution | `algorithm.py::solve_cmsa` | Updates `best` evaluating min across constructed and restricted solutions. |

### Decoupling of Fixed Modes in Construct

The codebase provides four orthogonal parameters in [`build_stage_model`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/formulation/stage_based.py):
1. `num_stages`: sets compact stage count $K = |C_{\text{truck}}| + 2$.
2. `fixed_truck_customers`: forces $\phi_h = 0$ for truck customers without locking route order.
3. `fixed_drone_customers`: forces $\phi_h = 1$ and $X_h^k = 0$ for drone customers.
4. `fixed_truck_route`: locks the exact sequence of truck nodes.

In `construct_solution()`, flexible truck route is evaluated first, enabling the truck to discover reordered routes to better coordinate with the drone. If subproblem budget is exhausted, fixed-route sequence is utilized, followed by greedy heuristic and all-truck certified fallbacks.

---

## Section 4 — Experimental Setup & Solver Transparency

1. **Solver differences**: The paper conducted experiments using commercial **IBM ILOG CPLEX 22.11** with `MIPEmphasis = 5` (Feasibility priority), 8 threads on an AMD Ryzen Threadripper PRO 5975WX with 252 GB RAM, and $t_{\text{MIP}} = 15\text{s}$. This reproduction utilizes **HiGHS** (via SciPy `milp`) with `mip_rel_gap = 0.02` in CMSA as an open-source alternative requiring no proprietary licenses, while preserving all 55 constraint equations.
2. **Benchmark instances (Table 3)**: The paper randomly generated 40 instances following Agatz et al.'s protocol ($n \in \{20, 30, 40, 50\}$, 10 instances each, uniform $[0, 100]^2$, depot $(50, 50)$, maxradius = 200%, truck speed 1, drone speed 2) but did not publish exact coordinate files or random seeds. Our 40 instances (generated via seeds 1–10) are an independent synthetic reproduction under the identical Agatz specification; objective values are statistical cross-instance comparisons rather than instance-by-instance matches.
3. **Model sizing (Table 4)**: Table 4 in the paper explicitly reports post-presolve dimensions from CPLEX 22.11 (`reported by CPLEX`), where fixed variables and redundant constraints are eliminated. In SciPy/HiGHS, we measure both raw static matrix dimensions and compact active subproblem dimensions (`free_variables`, `fixed_zero_variables`, `fixed_one_variables`, `active_constraints_before_presolve`, `active_nonzeros_before_presolve`), confirming that `age_limit=2` strictly and monotonically produces a smaller search space than `age_limit=5`.
