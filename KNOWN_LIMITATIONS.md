# Known Limitations and Methodological Disclosures

This document outlines the methodological differences, environment constraints, and reproducibility boundaries between this repository and the original publication:

> **"A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem"**  
> *Đức Minh Vũ, Minh Hoàng Hà, et al. (VIASM / NAFOSTED)*

---

## 1. Solver Backend: Commercial CPLEX vs Open-Source HiGHS

### Paper Protocol
- The authors used **IBM ILOG CPLEX 22.11** configured with:
  - `threads = 8`
  - `MIPEmphasis = 5` (Feasibility priority)
  - Time limit: 15s (restricted MIP) or 3600s/7200s (exact)

### Implementation in this Repository
- **Modular Solver Architecture (`fstsp.solver`)**:
  - `CplexBackend`: A genuine implementation using IBM ILOG CPLEX (`cplex` Python library). Passes the exact model objective, sparse constraint matrix, variable types (`B`, `C`), bounds, senses, and parameters.
  - `HighsBackend`: Open-source alternative using `scipy.optimize.milp` (HiGHS).
- **Strict Solver Fidelity**:
  - If a user configures `--solver cplex` on an environment without CPLEX installed, the system **strictly raises `CplexNotAvailableError`**. It does **NOT** silently fall back to HiGHS.
  - HiGHS is only executed when explicitly selected (`--solver highs` or configured in YAML).
- **Table 4 Presolve Metrics**:
  - CPLEX provides internal statistics for post-presolve variables and constraints (`c.solution.progress.get_num_variables()`).
  - SciPy's HiGHS wrapper does not expose post-presolve matrix dimensions. Under HiGHS, Table 4 reports the **active model dimensions before solver presolve** (`active_constraints_before_presolve`, `n_variables`, etc.), while `presolved_variables` is recorded as `None` with `presolve_metrics_available: false`.

---

## 2. Benchmark Instances: Synthetic 40 Instances vs Unpublished Author Instances

### Paper Table 3
- The authors evaluated Table 3 on 40 newly generated instances:
  - Sizes: $n \in \{20, 30, 40, 50\}$
  - 10 instances per size
  - Uniform coordinates in a $100 \times 100$ area
  - Truck speed $v_t = 1.0$, Drone speed $v_d = 2.0$
  - Service times $s_L = 1.0$, $s_R = 1.0$

### Implementation Disclosure
- The authors' original coordinate files and random seeds for these 40 instances were not published in the paper or public repositories.
- Consequently, this repository generates an **independent synthetic benchmark set of 40 instances** strictly adhering to the paper's generator specification (seeds 1 to 10 for each size $n \in \{20, 30, 40, 50\}$).
- While aggregate trends, relative gaps, and scaling behaviors can be compared directly, instance-by-instance objective values are independent reproductions.

---

## 3. Computational Budgets: Paper Protocol vs Smoke Tests

### Official Paper Budgets (`configs/paper_protocol.yaml`)
- **Table 1 & Table 2 (Agatz Benchmarks)**: Exact time limit = **3600 seconds (1 hour)**.
- **Table 3 Exact**: Time limit = **7200 seconds (2 hours)**.
- **Table 3 CMSA**: Total wall-clock time = **1800 seconds (30 minutes)**, restricted MIP = **15 seconds**, age limit = **2**.

### Rapid Verification Protocol (`configs/smoke_test.yaml`)
- For local testing, continuous integration, and Kaggle smoke tests, short runs (e.g. 10s to 45s) are used.
- Any run with a budget below 1800s is **explicitly tagged as `smoke_test` or `smoke_45s`** in CSV results, JSON metadata, and console logs.
- Short runs are never represented as full reproductions of the paper's 30-minute CMSA experiments.

---

## 4. Construction Heuristic (Algorithm 1 Implementation Details)

- The paper outlines the construction step at an architectural level: sample a subset of truck customers, solve MTZ TSP, and integrate drone customers using a stage-based formulation with $K = |C_{\text{truck}}| + 2$ stages.
- Because low-level details (tie-breaking, fallback mechanics when subproblem times out) were not published, this codebase implements:
  1. **Flexible Stage-Based Integration**: Allows truck customer re-ordering within the fixed stage count to coordinate with drone sorties.
  2. **Fixed-Route Stage-Based Fallback**: Locks the MTZ TSP tour sequence if flexible solve times out.
  3. **Greedy Heuristic Fallback**: Ensures algorithm robustness without stalling.
  4. **All-Truck Construction**: Safe, certified incumbent guaranteeing CMSA feasibility.
- The active method is tracked per iteration under `construction_method` and `stage_integration_mode`.

---

## 5. Table 3 Protocol Completion Gating

- An execution run is evaluated as `COMPLETE` only when all 40 instances across $n \in \{20, 30, 40, 50\}$ are solved under the full paper budget (1800s), utilizing CPLEX, certified by the zero-tolerance physical validator, without duplicates or smoke test contamination.
- Any local, CI, or partial execution (e.g. running 12 instances, or running under HiGHS, or running short budgets) is programmatically marked `PARTIAL` (e.g. `Completed: 12/40 | Missing: 28`) or `smoke_test`, preventing premature claims of full paper reproduction.

