# Changelog

## 0.4.1 — paper methodology alignment & audit hardening

- **Construct Stage-based Integration**: Replaced heuristic greedy consecutive-edge assignment with the paper's 2-index stage-based MILP subproblem using fixed truck tour and fixed stages $K = |C_{\text{truck}}| + 2$, enabling non-consecutive multi-stage drone sorties in Construct. Consecutive assignment is retained as a robust defensive fallback.
- **Strict Algorithm 1 Age Adaptation**: Synchronized `AgeManager.adapt()` with Lines 11–18 of Algorithm 1 by removing the `protected` set bypass. All active components are incremented by 1 at the end of each iteration, eliminating the 1-iteration lifespan bias.
- **Model Sizing Transparency (Table 4)**: Added `n_active_variables` ($UB > 0$) and `n_fixed_zero_variables` ($UB \le 0$) metrics to solution metadata, enabling rigorous pre-presolve search space comparisons between `age_limit=2` and `age_limit=5`.
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
