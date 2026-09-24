# Changelog

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
