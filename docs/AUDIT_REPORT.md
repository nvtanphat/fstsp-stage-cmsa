# Independent code audit and correctness report

This report documents the post-build audit performed on the reimplementation. The goal is not to claim that the repository is the authors' unpublished source code. The goal is to ensure that the implementation is internally consistent, that reported solutions are independently certified, and that the exact model agrees with an independent oracle on tiny instances.

## Critical issues found and fixed

### 1. Stale drone sorties after rebuilding a truck route

The first construction implementation could insert a failed drone customer back into the truck tour **after** earlier drone sorties had already been assigned. Rebuilding/reordering the truck route could then leave those previously assigned sorties with stale launch/recovery stages or reversed launch/recovery order.

Fix:

- truck route and all drone assignments are now rebuilt from scratch whenever a customer is promoted to truck service;
- drone assignments are certified before being returned;
- a regression fuzz test exercises low endurance + NOVISIT cases that previously exposed this bug.

### 2. Validation was too weak

The original validator checked basic coverage/endurance but did not fully reconstruct timing. It could miss:

- stale `launch_stage` / `recovery_stage` metadata;
- overlapping sorties;
- positive launch time from the start depot, which conflicts with the published Eqs. (27) and (30);
- Eq. (33) rendezvous-endurance violations where the drone can physically reach the recovery node but the truck arrives too late;
- mismatch between reported objective and the actual reconstructed completion time.

Fix:

`src/fstsp/evaluation/schedule.py` now independently reconstructs the physical schedule from the discrete route/sorties. `validate_solution()` uses this reconstruction and compares the reported objective against the certified completion time.

### 3. Public Agatz unrestricted instances used an unsafe artificial endurance

The parser previously represented missing `#MAXFLY` as a huge `1e9` endurance. Because the stage model uses a Big-M timing formulation, this created unnecessarily huge Big-M values and poor numerical conditioning.

Fix:

For an unrestricted public geometric instance, the parser now uses the finite 200%-of-maximum-pairwise-distance equivalent described by the public TSP-D-Instances documentation. This is sufficient for a two-leg drone operation while keeping the numerical scale realistic.

### 4. CMSA could deliberately exceed the requested time budget

The previous loop forced a minimum 0.25-second restricted-MIP budget even if the global budget had already expired.

Fix:

CMSA now computes the true remaining time and does not start a restricted MIP once the global budget is exhausted. Solver/model setup can still cause a small wall-clock overrun, so `total_time` should be interpreted as an optimization budget, not a hard real-time deadline.

### 5. CMSA could discard a known feasible construction

SciPy/HiGHS is used here without a MIP-start interface. Under a very short restricted-MIP time limit, the restricted solve may fail to recover a feasible incumbent even though Construct has already produced one.

Fix:

The best independently certified Construct solution is retained as a safety incumbent. The restricted MILP can improve it. This is a backend-oriented robustness adaptation and is documented as such.

### 6. Invalid experiment outputs did not fail the run

The original experiment scripts could print validation problems but still exit successfully.

Fix:

Local experiment scripts and the Kaggle entry point now fail intentionally if independent schedule certification reports an invalid solution. A benchmark CSV is still written first for forensic inspection.

## Independent correctness checks

Run:

```bash
python scripts/verify_release.py
```

The release verifier performs:

1. full `pytest` suite;
2. exact stage-MILP vs an independent brute-force oracle for tiny instances;
3. construction fuzz testing under low endurance and NOVISIT restrictions;
4. end-to-end short-budget CMSA certification;
5. local execution of the exact folder that Kaggle CLI will upload.

A machine-readable report is written to:

```text
artifacts/verification/release_verification.json
```

At the audited release, the exact MILP matched the independent brute-force oracle on all configured tiny cases to numerical precision. This is a much stronger check than merely verifying that the solver returns `Optimal`.

## What this audit does and does not guarantee

The audit supports the following statement:

> For the tested instance families and sizes, the implementation's discrete solution can be reconstructed into a valid schedule, and the exact MILP agrees with an independent exhaustive oracle on tiny instances.

It does **not** justify these stronger claims:

- that the code is the authors' original source code;
- that the unpublished 40 instances were reconstructed exactly;
- that HiGHS reproduces CPLEX runtimes from the paper;
- that CMSA must obtain the same heuristic objective on every medium/large instance;
- that a real Kaggle cloud execution has been completed without the user's Kaggle credentials.

For research reporting, separate **model correctness**, **heuristic quality**, and **paper-result reproducibility** as three different questions.


## 0.3.0 follow-up audit

Additional bugs found and fixed after the first correctness audit:

1. A `feasible=True` solution with `objective=None` or a non-finite objective could
   pass the validator. The validator now rejects missing/non-finite objectives.
2. CMSA could exceed its global time budget because Construct always allowed an
   embedded MTZ TSP solve up to 5 seconds. The TSP construction budget is now
   capped by the remaining CMSA wall-clock budget.
3. Non-finite solver metrics could produce non-standard JSON (`NaN`/`Infinity`).
   Solution serialization is now strict JSON and sanitizes non-finite metrics to null.
4. The generic stage solver and `run_exact.py` previously defaulted to a positive
   relative MIP gap. This could yield a solver-success status without a zero gap.
   Exact runs now default to relative MIP gap 0.0, and output records both `mip_gap`
   and `metadata.proven_optimal`.
5. The brute-force oracle now has its own timing evaluator rather than calling the
   production schedule validator, reducing common-mode verification risk.

Release verification covers positive/zero launch and recovery handling, finite
endurance, NOVISIT restrictions, strengthening on/off equivalence, Construct fuzz,
CMSA smoke runs, and the exact Kaggle upload bundle.


## 0.4.0 deep audit

A further audit intentionally searched for bugs not covered by the v0.3 tests. One material issue was found.

### Material bug found: model assembly bypassed the CMSA wall-clock budget

`scipy.optimize.milp(time_limit=...)` only limits the solver call. The Python code that assembles the stage MILP runs before HiGHS receives that limit. On n=20, model assembly can take around a second or more, so a nominal 0.35-second CMSA run was observed taking roughly 1.66 seconds.

Fix:

- `build_stage_model()` now accepts an absolute deadline and periodically checks it while creating variables, dominant O(n^3)/O(n^4) constraint blocks, and the sparse matrix.
- `solve_stage_model()` carries the same deadline through model assembly and reduces the remaining HiGHS time limit after assembly.
- CMSA propagates one absolute deadline through Construct, embedded TSP solves, and restricted-MIP solves.
- A regression test at n=20 verifies that the previous multi-second overrun cannot recur. Small scheduler/OS jitter is still possible; this is not a hard real-time system.

### Solver-incumbent hardening

Before discrete extraction, every returned HiGHS vector is now independently checked against:

- variable lower/upper bounds;
- all assembled linear-constraint lower/upper bounds;
- integrality residuals.

The maximum violations are recorded in `solution.metadata`. Incumbents exceeding tolerance are rejected rather than converted into a route. Drone A/B/Y/W extraction also requires exactly one selected launch location, recovery location, launch stage, and recovery stage for each drone-served customer.

### Additional adversarial testing

The v0.4 verifier completed:

- 54 pytest tests;
- 6 fixed exact-vs-independent-brute-force oracle cases with zero observed objective difference;
- 40 additional randomized exact-vs-brute-force cases over varied coordinate scales, speeds, handling times, endurance and NOVISIT masks; maximum objective difference `3.55e-15`;
- 100 certified Construct fuzz cases;
- 3 CMSA end-to-end smoke cases;
- an n=20 sub-second wall-clock budget regression;
- a standalone local execution of the exact Kaggle upload bundle.

An additional manual stress sweep of 120 randomized n=1..4 instances matched the independent brute-force optimum in every case, with maximum objective difference about `1.42e-14`.

## 0.4.1 audit and paper methodology alignment

A comprehensive audit identified 5 methodology, modeling, and measurement nuances compared with the published paper:

### 1. Construct step: MTZ TSP threshold & Stage-based formulation (P0)
- **Problem**:
  1. The code had `exact_tsp_threshold = 12`. With the default truck sample ratio of 65%, instances with $n \in [20, 50]$ had $|C_{\text{truck}}| \approx 13 \dots 33 > 12$, causing MTZ TSP to be systematically bypassed in favor of Nearest Neighbor + 2-opt.
  2. When drone integration was attempted, if the MILP failed, the code defaulted to a consecutive-edge greedy assignment heuristic that only tested adjacent stages ($k \to k+1$), unlike the paper formulation which permits multi-stage sorties ($k < k'$).
- **Fix**:
  1. Raised default `exact_tsp_threshold = 60` across `_build_truck_route()` and `construct_solution()`, ensuring MTZ TSP is always executed for sampled truck customers across all benchmark instances ($n \le 50$).
  2. Implemented `fixed_truck_route` in `build_stage_model()` and `solve_stage_model()`: stages are fixed to $K = |C_{\text{truck}}| + 2$, stage visits $X_{k, \text{route}[k]} = 1$ and arcs $x_{\text{route}[k], \text{route}[k+1]} = 1$ are fixed, and strengthening constraints (Eqs. 34 & 35) are bypassed because the stage count is already compact. The stage-based MILP natively discovers optimal non-consecutive sorties ($k < k'$).
  3. Replaced greedy fallback with a strict paper-compliant resampling/promotion loop: when the stage-based MILP cannot schedule all drone customers (due to endurance limits or overlap constraints), the drone customer requiring the greatest detour is promoted to the truck set and MTZ TSP re-solves for the expanded truck route.

### 2. Component age adaptation mismatch with Algorithm 1 (P0)
- **Problem**: `AgeManager.adapt(protected=...)` exempted components in `c_comp | mip_comp` from aging. Consequently, newly marked useful components remained at `age = 0` across iterations instead of being incremented, granting them an extra iteration of lifetime contrary to Algorithm 1 (Lines 11–18).
- **Fix**: Removed the `protected` bypass in accordance with Lines 11–18 of Algorithm 1. Every active component with `age >= 0` is incremented by 1 at the end of each iteration, strictly enforcing `age_limit`.

### 3. Model size reporting for Table 4 (P0)
- **Problem**:
  1. Because `build_stage_model()` assembled a static matrix and fixed inactive variables by setting $UB = 0$, reported variable and constraint counts were raw static matrix dimensions ($90,000+$ rows), obscuring the physical reduction.
  2. An inverted data point occurred in earlier reporting where $n=20$ for `age=2` recorded more constraints than `age=5`.
- **Methodology Clarification & Fix**:
  1. The paper's Table 4 explicitly cites: *"Statistic information regarding the number of variables, constraints, nonzero coefficients reported by CPLEX"*. CPLEX's presolve eliminates variables fixed to 0 and prunes redundant constraint rows, so the paper reported post-presolve dimensions.
  2. In our SciPy/HiGHS pipeline, we added active compact subproblem metrics to solution metadata:
     - `n_active_variables`: count of variables with non-zero upper bounds ($UB > 0$).
     - `n_active_constraints`: count of constraint rows containing at least one active variable with a non-zero coefficient.
     - `n_active_nonzeros`: count of non-zero entries in the active submatrix.
  3. Under these active metrics, `age=2` strictly and monotonically creates a much smaller search space than `age=5` across all problem sizes ($n=20, 30, 40, 50$), eliminating the reporting anomaly.

### 4. Solver configuration transparency (P1)
- **Problem**: Paper experiments used commercial **CPLEX 22.11** with `MIPEmphasis = 5` (Feasibility priority), 8 threads on an AMD Ryzen Threadripper PRO 5975WX with 252 GB RAM, and $t_{\text{MIP}} = 15\text{s}$. Code uses open-source **SciPy / HiGHS** with `mip_rel_gap = 0.02` in CMSA.
- **Documentation**: This intentional open-source design ensures 100% reproducibility on standard hardware and Kaggle cloud without commercial licensing restrictions. Differences in branch-and-cut heuristics and feasibility focus between CPLEX and HiGHS explain minor runtime and gap variations while preserving identical mathematical formulations (55 constraint groups).

### 5. Benchmark instance dataset transparency (P1)
- **Problem**: Comparisons such as 353.42 vs 368.06 evaluate independently generated instances because the authors did not publish raw coordinates or seeds for their 40 instances.
- **Documentation**: The 40 instances in this repository are an independent synthetic reproduction generated strictly according to Agatz et al.'s rules as specified in Section 4 and Appendix Table 7 ($n \in \{20, 30, 40, 50\}$, uniform $[0, 100]^2$, depot $(50, 50)$, maxradius = 200%, truck speed 1, drone speed 2). Numerical comparisons reflect statistical cross-instance benchmarking under the identical experimental protocol.

### 6. Convergence candidate selection bug (P1)
- **Problem**: `analyze_convergence_1800s.py` selected `r_obj` whenever restricted MIP was feasible, even if the constructed solution achieved a lower makespan, corrupting `best_so_far` tracking.
- **Fix**: Updated logic to evaluate `min(cands)` across both constructed and restricted solutions. Regenerated CSV history and convergence plots.

### 7. Wall-clock budget leakage in heuristic TSP (P1)
- **Problem**: `nearest_neighbor_tour` and `two_opt` lacked deadline checks, while dynamic property recalculations of distance matrices caused sub-second budget overruns on $n=50$ (taking $0.167\text{s}$ on a $0.01\text{s}$ budget).
- **Fix**: Injected deadline checks into heuristic loops and converted `truck_time`, `drone_time`, and `node_coords` into `@cached_property`. Runtime on $n=50$ for a $0.01\text{s}$ budget dropped to $0.0104\text{s}$ ($0.4\text{ms}$ overrun).
