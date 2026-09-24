# Changelog

## 0.4.3 — modular solver backends, CPLEX integration, config synchronization & Table 4 precision

- **P0 — Modular Solver Hierarchy (`fstsp.solver`)**:
  - Implemented `SolverBackend` abstraction with `CplexBackend` (using real `cplex.Cplex` API) and `HighsBackend` (using `scipy.optimize.milp`).
  - Strict absence handling: choosing `solver_backend='cplex'` on an environment without IBM ILOG CPLEX raises `CplexNotAvailableError` and does NOT fall back silently to HiGHS.
  - Implemented CPLEX parameters: `threads` (8), `mip_emphasis` (5, feasibility), `timelimit`, `mipgap`, and extracted real CPLEX progress metrics (presolved variables, presolved constraints, node counts).
- **P0 — Centralized Configuration Loader (`fstsp.config`)**:
  - Created [`src/fstsp/config.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/config.py) and [`configs/smoke_test.yaml`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/configs/smoke_test.yaml).
  - Explicitly separated paper protocol (Table 1/2: 3600s, Table 3 Exact: 7200s, Table 3 CMSA: 1800s, restricted MIP: 15s, age: 2) from smoke test protocol (`smoke_45s`).
  - Propagated config through `experiments/run_exact.py`, `experiments/run_cmsa.py`, and `kaggle/run_experiment.py`.
- **P0 — Fully Fixed Table 4 Measurement & Isolation**:
  - Completely removed erroneous fallback from variable counts to `active_components`.
  - Decomposed model metrics: `n_variables`, `n_fixed_zero_variables`, `n_fixed_one_variables`, `n_free_variables`, `n_constraints`, `n_nonzeros`, `presolved_variables`, `presolved_constraints`, `presolved_nonzeros`.
  - Explicitly labeled measurement methodology: CPLEX post-presolve dimensions vs HiGHS pre-presolve active dimensions.
- **P1 — Table 3 Budget & Dataset Standardization**:
  - Strictly banned mixing 45s smoke runs with 1800s paper runs.
  - Fully disclosed independent 40-instance synthetic reproduction versus unpublished author instances in [`KNOWN_LIMITATIONS.md`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/KNOWN_LIMITATIONS.md).
- **Added Comprehensive Unit Test Suite**:
  - Added [`tests/unit/test_cplex_backend.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/tests/unit/test_cplex_backend.py), [`tests/unit/test_solver_backend_selection.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/tests/unit/test_solver_backend_selection.py), [`tests/unit/test_paper_config_propagation.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/tests/unit/test_paper_config_propagation.py), [`tests/unit/test_table3_budget_consistency.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/tests/unit/test_table3_budget_consistency.py), and [`tests/unit/test_table4_metrics.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/tests/unit/test_table4_metrics.py).
  - Verified total automated test count: **79 / 79 tests collected**.

## 0.4.2 — mathematical formulation proof, construct decoupling & audit hardening

- **Mathematical Proof & Equation (35) Fix**: Corrected constant $K$ in Equation (35) ($\sum_k k X_E^k + \sum_h \phi_h = N + 2$) to strictly reference the total original customer count $N + 2$ (`instance.n + 2`), proving consistency for both full formulation and compact stage subproblems ($K_{\text{sub}} < N + 2$). Enabled Equations (34) and (35) unconditionally under `strengthen=True`.
- **Orthogonal Decoupling of Construct Parameters**: Refactored `build_stage_model()` and `solve_stage_model()` to cleanly distinguish four independent mathematical choices:
  - `num_stages`: sets compact stage horizon ($K = |C_{\text{truck}}| + 2$).
  - `fixed_truck_customers`: fixes $\phi_h = 0$ for truck customers without locking route order.
  - `fixed_drone_customers`: forces $\phi_h = 1$ and $X_h^k = 0$ for drone customers.
  - `fixed_truck_route`: locks the exact sequence of truck nodes.
- **Flexible Route Exploration in Construct**: `_integrate_drone_customers_stage_based` now evaluates flexible truck routes first, allowing the truck to reorder customer stops to better coordinate with the drone, falling back to locked route sequence if the flexible subproblem times out.
- **MTZ TSP Status Reporting**: Added `TSPResult` dataclass to `src/fstsp/algorithms/tsp/mtz.py`, explicitly distinguishing `OPTIMAL` (proven optimal), `FEASIBLE` (incumbent found), `TIME_LIMIT`, `INFEASIBLE`, and `ERROR`, validating extracted tours against subtours.
- **Explicit Fallback Tracking**: Every constructed solution strictly records its actual generation method in `solution.status` and `solution.metadata["construction_method"]`: `"stage_based_milp"`, `"heuristic_fallback"`, or `"all_truck_fallback"`, preventing any silent misattribution.
- **Table 4 Model Sizing Separation & Bug Fix**: Separated model metrics into `original_variables`, `fixed_zero_variables`, `fixed_one_variables`, `free_variables`, `active_constraints_before_presolve`, `active_nonzeros_before_presolve`, and `presolved_variables` (null for HiGHS). Fixed counting bug where fixed-one variables were misclassified as fixed-zero.
- **Official Paper Protocol Configuration**: Added [`configs/paper_protocol.yaml`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/configs/paper_protocol.yaml), differentiating Table 1/2 Exact limit (3600s), Table 3 CPLEX Exact limit (7200s), Table 3 CMSA limit (1800s), and $t_{\text{MIP}} = 15\text{s}$.
- **10 New Audit Regression Tests**: Added [`tests/regression/test_paper_audit_fixes.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/tests/regression/test_paper_audit_fixes.py), bringing total automated test count to **65 / 65 passed**.


- **Construct MTZ TSP & Stage-based Integration**: Set default `exact_tsp_threshold = 60` (previously 12), ensuring MTZ TSP is executed for all benchmark sizes up to $n=50$. Formulated exact drone customer integration via the 2-index stage-based MILP with fixed truck route and $K = |C_{\text{truck}}| + 2$ stages, enabling native discovery of non-consecutive multi-stage sorties ($k < k'$).
- **Resampling/Promotion Loop**: Replaced consecutive greedy assignment with a paper-compliant resampling/promotion loop: unserviceable drone customers are promoted to the truck route and MTZ TSP re-runs for the expanded set.
- **Strict Algorithm 1 Age Adaptation**: Synchronized `AgeManager.adapt()` with Lines 11–18 of Algorithm 1 by removing the `protected` set bypass. All active components are incremented by 1 at the end of each iteration, strictly enforcing component lifespan.
- **Model Sizing Transparency (Table 4)**: Added `n_active_variables`, `n_active_constraints`, and `n_active_nonzeros` to solution metadata. Clarified Table 4 CPLEX post-presolve dimensions vs open-source HiGHS active subproblem dimensions, proving `age_limit=2` strictly produces a smaller search space than `age_limit=5`.
- **Solver & Dataset Transparency**: Documented solver configuration differences (CPLEX 22.11 with MIPEmphasis 5 vs open-source SciPy/HiGHS with `mip_rel_gap=0.02`) and dataset independence (40 newly generated instances under Agatz protocol vs unpublished original seeds).
- **Convergence History Bug Fix**: Fixed candidate selection logic in `analyze_convergence_1800s.py` to evaluate `min(constructed, restricted)` so sub-optimal restricted MIPs cannot overwrite better constructed solutions.
- **Absolute Deadline Enforcement**: Added deadline parameters and loop timeout checks to `nearest_neighbor_tour` and `two_opt`; converted `truck_time`, `drone_time`, and `node_coords` to `@cached_property` on `FSTSPInstance`. Sub-second budget overrun on $n=50$ dropped from $167\text{ms}$ to $0.4\text{ms}$.
- **Resilient Test Collection**: Guarded `test_kaggle_candidate_detection.py` against missing `kaggle/` folder in clean git clones.
- Added regression test `test_cmsa_tight_deadline_large_instance` (55 tests passing).

## 0.4.0 — deep audit / deadline hardening

- Added direct residual checks on every HiGHS incumbent: variable bounds, all linear rows, and integrality before solution extraction.
- Replaced permissive `argmax` drone extraction with exact one-hot extraction checks for A/B/Y/W variables.
- Fixed a real CMSA wall-clock bug: stage-model **assembly** was outside the solver time limit and could overrun a 0.35 s budget by more than 1 second at n=20. Model construction now accepts an absolute deadline and aborts safely when expired.
- Propagated the CMSA deadline into repeated Construct/TSP subproblems; after deadline exhaustion Construct switches to a cheap certified fallback instead of launching another MTZ solve.
- Hardened domain validation against NaN/Infinity and empty-customer instances.
- Hardened TSP-MTZ public input validation.
- Extended strict JSON serialization to NumPy arrays, sets and Paths.
- Fixed Kaggle Agatz discovery so `maxradius`/`novisit` can be recognized from parent directories as well as filenames.
- Added randomized exact-vs-independent-brute-force regression tests and deadline regression tests.
- Release verifier now includes 40 randomized independent-oracle cases plus a dedicated CMSA wall-clock regression.

## 0.3.0

- Fixed feasible-solution validation accepting a missing/non-finite objective.
- Fixed CMSA construction potentially violating the overall wall-clock budget via a fixed 5 s embedded TSP solve.
- Added bounded construction-TSP budget and input validation for solver gap/time arguments.
- Sanitized non-finite solver metrics so `solution.json` is strict JSON.
- Added strengthening-equivalence and objective-validation regression tests.
- Exact stage solves now default to zero relative MIP gap and report `metadata.proven_optimal`; solver-success text alone is not treated as a proof.


## 0.2.0 — correctness audit release

- Fixed stale drone sortie/stage assignments after truck-route rebuilds in Construct.
- Added independent schedule reconstruction and objective certification.
- Added Eq. (33)-style rendezvous endurance validation and sortie overlap checks.
- Stage-MILP solutions now retain raw solver objective but report certified completion time.
- Added independent brute-force tiny-instance oracle and exact-vs-oracle regression tests.
- Replaced artificial unrestricted Agatz endurance `1e9` with finite 200% geometric equivalent.
- Fixed CMSA global time-budget handling.
- Retain best certified Construct solution as safety incumbent for HiGHS backend.
- Experiment and Kaggle runs now fail if independent certification fails.
- Added synthetic and attached-Agatz modes to Kaggle script-kernel bundling.
- Updated Kaggle CLI documentation against current Sep 2026 official command docs.
- Added `scripts/verify_release.py` and machine-readable release verification report.

## 0.1.0

Initial paper-based reimplementation.
