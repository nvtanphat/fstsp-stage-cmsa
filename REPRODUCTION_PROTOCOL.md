# FSTSP Paper Reproduction Protocol

This document provides exact instructions to run, reproduce, and verify the experimental findings of:

> **"A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem"**  
> *Đức Minh Vũ, Minh Hoàng Hà, et al. (VIASM / NAFOSTED)*

---

## 1. Environment & Solvers

### 1.1 Requirements
- Python $\ge$ 3.11
- Core dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`
- Test dependencies: `pytest`

### 1.2 Solver Backends
The codebase supports two solver backends via `fstsp.solver`:
1. **Commercial Paper Benchmark (`--solver cplex`)**:
   - Requires IBM ILOG CPLEX Optimization Studio (e.g. 22.11) with a valid license and the `cplex` Python package.
   - Executes with 8 threads and `MIPEmphasis = 5` (Feasibility priority) per Section 4.
2. **Open-Source Default (`--solver highs`)**:
   - Out-of-the-box open-source solver using `scipy.optimize.milp` (HiGHS).
   - Requires no commercial licenses.

> [!IMPORTANT]
> The codebase strictly avoids silent solver fallback. If `--solver cplex` is requested on an environment without CPLEX installed, it raises `CplexNotAvailableError` immediately to guarantee scientific transparency.

---

## 2. Configuration Profiles

Configurations are managed in YAML files under `configs/`:

| Config File | Target Usage | CMSA Budget | Exact Budget | Protocol Label |
|---|---|---|---|---|
| `configs/paper_protocol.yaml` | Official paper reproduction | 1800s (30 min) | 7200s (2 hrs) | `paper_1800s` |
| `configs/smoke_test.yaml` | CI, quick verification, local testing | 10s–45s | 30s–45s | `smoke_45s` |

---

## 3. Running Experiments

### 3.1 Local Exact Model Run
Solve a single instance using the 2-index stage-based MILP:
```bash
python experiments/run_exact.py --n 6 --seed 42 --solver highs --time-limit 30
```
With CPLEX (if licensed):
```bash
python experiments/run_exact.py --n 6 --seed 42 --solver cplex --time-limit 30 --threads 8 --mip-emphasis 5
```

### 3.2 Local CMSA Run
Solve an instance using CMSA (Algorithm 1):
```bash
python experiments/run_cmsa.py --n 20 --seed 42 --solver highs --protocol smoke
```
Full paper budget (30 minutes):
```bash
python experiments/run_cmsa.py --n 20 --seed 42 --solver highs --protocol paper
```

### 3.3 Kaggle / Full Experiment Suite
To run the automated suite reproducing Tables 1–4 and Figure 1:
```bash
# Smoke test mode (fast check)
FSTSP_PROTOCOL=smoke python kaggle/run_experiment.py

# Full paper protocol with HiGHS
FSTSP_PROTOCOL=paper FSTSP_SOLVER_BACKEND=highs python kaggle/run_experiment.py

# Official paper protocol with CPLEX (requires CPLEX license)
FSTSP_PROTOCOL=paper FSTSP_SOLVER_BACKEND=cplex python kaggle/run_experiment.py
```

---

## 4. Expected Artifacts

When experiments complete, results are exported to `artifacts/kaggle/` (or `/kaggle/working/`):
- `table1_reproduction.csv`: Maxradius benchmark summary ($n=10, 20$).
- `table2_reproduction.csv`: Novisit benchmark summary ($n=10$).
- `table3_reproduction.csv`: Comparison of Exact vs CMSA on 40 instances ($n=20, 30, 40, 50$).
- `table3_details.csv`: Per-instance objective, runtime, and improvement gap.
- `table4_reproduction.csv`: Constraints, variables, and nonzeros across `age=2` vs `age=5`.
- `table4_details.csv`: Detailed model dimensions with decomposed variable partitions.
- `route.png`: Certified route visualization.
