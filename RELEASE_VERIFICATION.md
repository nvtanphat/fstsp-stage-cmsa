# Release Verification Report

This document records the comprehensive verification checks for the FSTSP paper reproduction release.

---

## 1. Automated Test Suite Execution

Run command:
```bash
pytest -v
```

### Coverage & Verification Targets
- **Unit Tests (`tests/unit`)**:
  - `test_solver_backend_selection.py`: HiGHS and CPLEX backend factory instantiation, error handling for unsupported backends, and strict verification that CPLEX does **not** fall back silently to HiGHS.
  - `test_cplex_backend.py`: Verification that missing CPLEX raises `CplexNotAvailableError`, and real CPLEX integration solves models and reports native solver stats when licensed.
  - `test_paper_config_propagation.py`: Verification of official paper budgets (3600s, 7200s, 1800s, 15s) and separation from smoke test configs.
  - `test_table3_budget_consistency.py`: Strict check ensuring Table 3 runs distinguish `paper_1800s` from `smoke_45s`.
  - `test_table4_metrics.py`: Mathematical decomposition of variable types (`fixed_zero`, `fixed_one`, `free`) ensuring they sum to total variables, and verification that active components count never substitutes variable count.
  - `test_agatz_parser.py`: Geometry and parameter parsing for Agatz benchmark instances.
  - `test_age.py`: Algorithm 1 component aging without artificial protected set.
  - `test_cmsa_budget_args.py`: Wall-clock budget enforcement in CMSA iterations.
  - `test_instance_validation.py`: Verification of coordinate and parameter boundaries.
  - `test_schedule_validator.py`: Zero-tolerance physical timeline validator.
  - `test_tsp_mtz_validation.py`: MTZ TSP formulation checks and `TSPResult` status.
- **Integration Tests (`tests/integration`)**:
  - `test_bruteforce_oracle.py`: Numerical equivalence with exhaustive permutation brute-force oracle down to machine precision.
  - `test_equation35_invariance.py`: Verification that Equation (35) constant RHS equals $N+2$ across varied stage counts.

---

## 2. Mathematical Consistency Checks

### Equation (35) Invariance
$$\sum_k k X_E^k + \sum_h \phi_h = N + 2$$
The constant RHS is strictly $N+2$ (`instance.n + 2`), validated analytically and verified via automated tests across varied fixed-stage counts and customer subsets.

### Solver Backend Modularization
- `CplexBackend` implements the full IBM ILOG CPLEX API with `threads=8` and `mip_emphasis=5`.
- `HighsBackend` wraps `scipy.optimize.milp` for open-source reproducibility.
- No silent fallback: selecting `cplex` when unavailable raises `CplexNotAvailableError`.

---

## 3. Independent Schedule Certification

Every solution produced by the exact model or CMSA passes through `fstsp.evaluation.validator.validate_solution`:
- Truck path connectivity: contiguous tour starting at $0$ and ending at $N+1$.
- Drone sorties: single-customer deliveries launched from truck node $i$ at stage $k$ and recovered at truck node $j$ at stage $k'$.
- Non-overlapping sorties: no overlapping drone operations in time.
- Battery endurance: drone flight time plus handling times $\le D_d$.
- Makespan synchronization: arrival times match continuous timeline calculation.
