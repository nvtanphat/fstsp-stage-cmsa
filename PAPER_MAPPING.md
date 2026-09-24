# Mathematical Formulation and Algorithm Mapping

This document provides a line-by-line mapping between the mathematical formulation in the published paper:

> **"A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem"**  
> *Đức Minh Vũ, Minh Hoàng Hà, et al. (VIASM / NAFOSTED)*

and the corresponding implementation files in this repository.

---

## 1. Mathematical Formulation: Equations (1)–(55)

The mathematical model is implemented in [`src/fstsp/formulation/stage_based.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/formulation/stage_based.py).

| Equation(s) | Description | Implementation Function / Location |
|---|---|---|
| **(1)** | Objective function: Minimize makespan $d_E$ (completion time at end depot) | `build_stage_model`: Objective vector $c$ sets $c[d_E] = 1.0$, all other coefficients $0$. |
| **(2)–(6)** | Truck tour connectivity, stage transition, and depot constraints | Constraints connecting stage sequence $X_i^k$ to arc flows $x_{ij}$. |
| **(7)–(10)** | Stage assignment: exactly one node per stage, customer visited by truck at most once | Rows enforcing $\sum_{i} X_i^k = 1$ and $\sum_{k} X_i^k + \phi_i = 1$ for customers. |
| **(11)–(15)** | Precedence of stages and start/end depot positioning | Stage 0 pinned to $S$ ($X_S^0 = 1$), end depot pinned to stage $k_E$. |
| **(16)–(20)** | Drone launch ($A_{hi}$), recovery ($B_{hj}$), launch stage ($Y_h^k$), recovery stage ($W_h^k$) | Binary assignment and linkage constraints connecting drone sorties to truck stages. |
| **(21)–(23)** | Customer coverage: every customer served by either truck or drone | $\sum_k X_h^k + \phi_h = 1 \quad \forall h \in C$. |
| **(24)–(26)** | Stage timing and service delays: Big-M arrival/departure propagation | Big-M constraints linking departure time $d_k$ to arrival time $t_k$ and travel times. |
| **(27)–(30)** | Continuous sortie duration $Z_{kk'}$ and drone endurance $D_d$ | $Z_{kk'}$ continuous linearization, flight duration $\le D_d$, handling times $s_L, s_R$. |
| **(31)–(33)** | Non-overlapping drone sorties and launch/recovery rendezvous synchronization | Disjunctive constraints ensuring the single drone does not perform simultaneous flights. |
| **(34)–(40)** | Strengthening cuts / valid inequalities | Valid inequalities; Eq (35): $\sum_k k X_E^k + \sum_h \phi_h = N + 2$. |
| **(41)–(55)** | Variable bounds, non-negativity, and integrality | Variables defined via `integrality` array (1 for binary/integer, 0 for continuous) and `Bounds`. |

---

## 2. Metaheuristic: CMSA Algorithm 1

Implemented in [`src/fstsp/algorithms/cmsa/algorithm.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/algorithms/cmsa/algorithm.py).

| Paper Algorithm 1 Step | Implementation | File / Class |
|---|---|---|
| **Lines 1–3**: Initialization, best solution $S_{\text{bs}} \leftarrow \emptyset$, component ages $\text{age}(c) \leftarrow 0$ | Initializes `AgeManager(age_limit)`, `best = None`, `history = []`. | [`algorithm.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/algorithms/cmsa/algorithm.py) |
| **Line 5**: `GenerateSolution()` (Construction) | `construct_solution()`: samples truck customers, solves MTZ TSP, integrates drone customers with fixed stages $K = |C_{\text{truck}}| + 2$. | [`construct.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/algorithms/cmsa/construct.py) |
| **Lines 6–8**: Component merge & reset ages | `ages.mark_useful(c_comp)` resets ages of components in the constructed solution to 0. | [`age.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/algorithms/cmsa/age.py) |
| **Line 9**: `ApplyExactSolver()` (Restricted MIP) | `solve_stage_model()` with `active_components=active`, time limit $t_{\text{MIP}} = 15\text{s}$. | [`stage_based.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/formulation/stage_based.py) |
| **Line 10**: Incumbent update | If restricted solution is feasible and beats $S_{\text{bs}}$, update $S_{\text{bs}} \leftarrow S_{\text{mip}}$. | [`algorithm.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/algorithms/cmsa/algorithm.py) |
| **Lines 11–18**: Age adaptation & component elimination | `ages.adapt()`: increments ages of all active components; removes components with $\text{age} > \text{age}_{\text{limit}}$. Strictly follows paper without artificial protected sets. | [`age.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/algorithms/cmsa/age.py) |

---

## 3. Experimental Study: Section 4

| Experiment / Table | Paper Specification | Implementation Script |
|---|---|---|
| **Table 1** | Agatz maxradius instances ($n=10, 20$), exact stage MILP, 3600s budget | `run_paper_table1()` in [`kaggle/run_experiment.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/kaggle/run_experiment.py) |
| **Table 2** | Agatz novisit instances ($n=10$), exact stage MILP, 3600s budget | `run_paper_table2()` in [`kaggle/run_experiment.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/kaggle/run_experiment.py) |
| **Table 3** | 40 synthetic instances ($n=20, 30, 40, 50$), Exact (7200s) vs CMSA (1800s, $t_{\text{MIP}}=15\text{s}$, age=2) | `run_paper_table3()` in [`kaggle/run_experiment.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/kaggle/run_experiment.py) |
| **Table 4** | Constraints, variables, and coefficients for `age_limit=2` vs `age_limit=5` | `run_paper_table4()` in [`kaggle/run_experiment.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/kaggle/run_experiment.py) |
| **Figure 1** | Vehicle route visualizations for $n=30$ | `plot_solution()` in [`src/fstsp/visualization/route.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/visualization/route.py) |
