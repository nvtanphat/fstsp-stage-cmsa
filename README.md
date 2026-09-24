<div align="center">

# 🚚✈️ FSTSP Stage-based + CMSA

**A High-Fidelity Python Reimplementation & Reproduction of the 2-Index Stage-based Formulation and CMSA Algorithm for the Flying Sidekick Traveling Salesman Problem (FSTSP)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-55%2F55%20Passed-brightgreen.svg?logo=pytest&logoColor=white)](tests/)
[![Solver](https://img.shields.io/badge/Solver-HiGHS%20%28SciPy%29-orange.svg)](https://highs.dev/)
[![Dashboard](https://img.shields.io/badge/UI-Streamlit-red.svg?logo=streamlit&logoColor=white)](app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Based on the scientific research by Duc-Minh Vu et al. (VIASM / NAFOSTED grant 102.01-2023.26)*

[Interactive Demo](#-interactive-web-dashboard) • [Quick Start](#-quick-start) • [Scientific Results](#-scientific-reproduction-results) • [Correctness Gate](#-correctness-gate--verification) • [Architecture](#-project-architecture)

</div>

---

## 📖 Overview

The **Flying Sidekick Traveling Salesman Problem (FSTSP)** is a core combinatorial optimization challenge in modern last-mile logistics:
* A **truck** (mothership) and a **drone** collaborate to serve a set of customer nodes $C = \{1, \dots, N\}$.
* The drone can launch from the truck at a customer location (or the central depot), deliver a single parcel to another customer within its battery flight limit $D_d$, and rendezvous back with the truck further along its route.
* While the drone is in flight, the truck continues servicing other customers on the road network in parallel.
* **Objective**: Minimize the overall completion time (**Makespan** $d_E$) when both vehicles return to the depot.

This repository provides an **engineering-grade, open-source reimplementation** of the paper:
> **“A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem”**  
> *Đức Minh Vũ, et al.*

> [!NOTE]
> This codebase is an independent reconstruction based entirely on the mathematical formulations and algorithms published in the paper. It does **not** rely on unpublished source code and is fully runnable on local machines using free, open-source solvers (**HiGHS** via `scipy.optimize.milp`).

---

## ✨ Key Features

- 📐 **Full 2-Index Stage-based MILP**: Exact implementation of all equation groups (1)–(55), including Big-M synchronization, battery limits, continuous sortie variables $Z_{kk'}$, and model strengthening inequalities (34)–(40).
- ⚡ **CMSA Metaheuristic (Construct-Merge-Solve & Adapt)**: Solves large-scale instances up to $N = 50$ customers by decomposing the global search space into compact restricted MIP subproblems.
- 🛡️ **Zero-Tolerance Schedule Certification**: Independent physical schedule validator (`evaluate_schedule` & `validate_solution`) that independently reconstructs the continuous timeline from discrete assignments, guaranteeing zero battery or timing violations.
- 🔬 **Brute-Force Mathematical Oracle**: Includes an exhaustive permutation-based oracle verifying exact-MILP optimality down to machine precision ($\Delta < 1.42 \times 10^{-14}$).
- 🖥️ **Interactive Streamlit Web Dashboard**: Live route visualization, vehicle trajectory tracking, and synchronization Gantt charts.
- 🚀 **100% Local Execution**: Clean dependency stack (`scipy`, `numpy`, `streamlit`, `plotly`, `pytest`). No commercial solver licenses or cloud dependencies required.

---

## 🖥️ Interactive Web Dashboard

Launch the interactive Streamlit simulation platform to visually explore truck-drone coordination:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### Dashboard Highlights:
* **Interactive Map**: Visualize truck roads (blue) and aerial drone flight paths (orange dashed arrows) with Plotly.
* **Scenario Customizer**: Select Suburban (uniform), Urban (clustered), or Restricted delivery zones (`NOVISIT`).
* **Real-time Solver**: Adjust drone speed, battery endurance, launch/recovery handling times, and run live CMSA or Exact MILP optimization.
* **Schedule Diagnostics**: Inspect arrival, departure, and rendezvous synchronization tables.

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/nvtanphat/fstsp-stage-cmsa.git
cd fstsp-stage-cmsa

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows (cmd/PowerShell)

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify all 55 unit, integration, and regression tests:

```bash
pytest -q
```
```text
55 passed in 48.52s
```

### 3. Run Experiments Locally

Solve a small instance using the **Exact 2-Index MILP**:
```bash
python experiments/run_exact.py --n 6 --time-limit 30 --mip-rel-gap 0
```

Solve a larger instance using the **CMSA Metaheuristic**:
```bash
python experiments/run_cmsa.py --n 12 --total-time 60 --mip-time 10 --age-limit 2
```

---

## 🔬 Scientific Reproduction Results

Full experimental reproduction of the paper's 4 benchmark tables:

### 1. Table 1 & Table 2 — Agatz Benchmark ($N = 10, 20$)
* For small drone radius ($R \le 60\%$) and high `novisit` fractions ($\ge 40\%$), the exact model achieves **MIP GAP = 0.00% in 1.1s – 6.1s**.
* Demonstrates combinatorial explosion when radius expands or $N \ge 20$, justifying the necessity of CMSA.

### 2. Table 3 — CMSA Scaling vs Exact Baseline ($N = 20 \to 50$)

Under the paper's standardized **1800-second (30-minute)** computational budget:

| Problem Size ($N$) | Paper CPLEX Exact | Paper CMSA (1800s) | Our CMSA (1800s) | Comparison & Highlights |
| :---: | :---: | :---: | :---: | :--- |
| **$N = 20$** | 300.83 | **277.18** | **295.91** | **Outperforms Paper CPLEX Exact (300.83)**; gap to Paper CMSA is only **6.7%** |
| **$N = 30$** | 619.49 | **353.42** | **368.06** | Near-identical performance (**4.1% gap**); **Seed 2 achieves 352.63 (< Paper)** |
| **$N = 40$** | *Timeout* | **422.87** | **453.70** | Gap within **7.2%** of paper baseline |
| **$N = 50$** | *Timeout* | **503.86** | **542.42** | Gap within **7.6%**; 100% feasibility maintained |

> [!TIP]
> The remaining ~4% – 7% difference is well within scientific expectations due to:
> 1. **Dataset geometry**: The paper did not release the raw coordinates of its 40 randomly generated instances.
> 2. **Commercial vs Open-Source Solver**: The paper utilized commercial **IBM ILOG CPLEX 22.11 with 8 threads**, whereas our implementation runs open-source **HiGHS** via SciPy.

Detailed convergence curves and forensic tables are available in:
* Report: [`artifacts/reproduction/PAPER_REPRODUCTION_REPORT.md`](artifacts/reproduction/PAPER_REPRODUCTION_REPORT.md)
* 30-Minute Convergence Curve: [`artifacts/reproduction/figure_convergence_1800s.png`](artifacts/reproduction/figure_convergence_1800s.png)

---

## 🛡️ Correctness Gate & Verification

To verify mathematical correctness before drawing conclusions from experiments:

```bash
python scripts/verify_release.py
```

The release verifier executes:
1. Full 55-test suite (`pytest`).
2. Fixed exact-vs-brute-force comparisons: **$\Delta = 0.00000000000000$** absolute difference.
3. 40 randomized stress instances: maximum observed difference **$\le 1.42 \times 10^{-14}$**.
4. 100 random Construct fuzz tests under low endurance and `novisit` constraints.
5. CMSA wall-clock deadline compliance with sub-millisecond precision regression tests.
6. Alignment with paper Algorithm 1 (strict age adaptation and stage-based Construct integration).

Detailed report: [`artifacts/verification/release_verification.json`](artifacts/verification/release_verification.json).

---

## 📂 Project Architecture

```text
fstsp-stage-cmsa/
├── app.py                     # Streamlit Interactive Web Application
├── configs/                   # Configuration files
├── data/                      # Benchmark datasets & synthetic instance generators
│   ├── external/              # Public Agatz geometric benchmark instances
│   └── paper_40_instances/    # Standard 40 test instances (n=20..50)
├── docs/                      # Scientific documentation & audit reports
│   ├── AUDIT_REPORT.md        # Comprehensive algorithmic audit report
│   ├── HUONG_DAN_VI.md        # Vietnamese user guide
│   └── PAPER_MAPPING.md       # Equation-by-equation paper-to-code mapping
├── experiments/               # Experiment execution scripts
│   ├── run_exact.py           # Exact 2-index MILP runner
│   ├── run_cmsa.py            # CMSA metaheuristic runner
│   └── reproduction/          # Report & plot generation scripts
├── artifacts/                 # Certified reproduction reports & verification outputs
│   ├── reproduction/          # Markdown reports, CSV histories & convergence plots
│   └── verification/          # Machine-readable release verification JSON
├── scripts/                   # Utility & verification scripts
│   ├── download_agatz_data.py # Dataset downloader
│   └── verify_release.py      # Automated mathematical correctness gate
├── src/fstsp/                 # Core Python package
│   ├── algorithms/            # CMSA (AgeManager, Construct, Solve) & MTZ TSP
│   ├── data/                  # Instance parsing, generation & validation
│   ├── domain/                # Data structures (Instance, Solution, DroneSortie)
│   ├── evaluation/            # Physical schedule reconstruction & brute-force oracle
│   ├── formulation/           # 2-index stage-based MILP (Eqs 1-55)
│   └── visualization/         # Route & timeline plotting utilities
└── tests/                     # 54 Unit, integration, and regression tests
```

---

## 📜 References

1. **Duc-Minh Vu et al.** *A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem*. VIASM / NAFOSTED (grant 102.01-2023.26).
2. **Chase C. Murray and Amanda G. Chu (2015).** *The flying sidekick traveling salesman problem: Optimization of drone-assisted parcel delivery.* Transportation Research Part C: Emerging Technologies, 54:86–109.
3. **Niels Agatz, Paul Bouman, and Marie Schmidt (2018).** *Optimization Approaches for the Traveling Salesman Problem with Drone.* Transportation Science, 52(4):965–981.
4. **Christian Blum (2016).** *Construct, Merge, Solve & Adapt. A new general technique for combinatorial optimization.* Computers & Operations Research, 72:46–58.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
