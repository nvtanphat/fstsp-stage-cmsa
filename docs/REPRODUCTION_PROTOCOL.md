# FSTSP Paper Reproduction Protocol

This document establishes the scientific reproduction protocol and audit standards for:

> **A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem**  
> *Đức Minh Vũ and collaborators (VIASM / NAFOSTED grant 102.01-2023.26)*

---

## 1. Mathematical Formulation Reference

The authoritative reference for all equations is Section 2 of the paper (Equations 1–55), implemented in [`src/fstsp/formulation/stage_based.py`](file:///D:/HOCsauvaufngdung/fstsp_audit_v04_clean/src/fstsp/formulation/stage_based.py).

### Core Constraint Blocks

| Category | Equations | Implementation Symbol | Mathematical Semantics |
|---|---|---|---|
| **Objective** | (1) | `c[d_E] = 1.0` | Minimize trip makespan ($d_E$). |
| **Truck Routing** | (2)–(5) | Depot arrival/departure | $X_S^1 = 1$, $X_S^k = 0$ ($k \ge 2$), $\sum_{k=2}^K X_E^k = 1$, $x_{Ei} = 0$. |
| **Connectivity** | (6)–(7) | Flow conservation | Degree balance between stages and routing arcs $x_{ij}$. |
| **Stage Exclusivity** | (8)–(9) | Stage synchronization | $\sum_i X_i^k \le 1$; $X_j^{k+1} + 1 - x_{ij} \ge X_i^k$. |
| **Drone Consistency** | (10)–(13) | Sortie definition | Exactly one launch ($A_h^i$), recovery ($B_h^i$), and sortie span ($Z_{kk'}$). |
| **Drone Endurance** | (14) | Flight endurance | Flight time $\tau'_{ih} + \tau'_{hi} \le (D_d - t_R)\phi_h$. |
| **Non-overlapping Sorties** | (15) | Sortie concurrency | $\sum_{k \le m, k' > m} Z_{kk'} \le 1$ for all cut stages $m$. |
| **Drone-Truck Forcing** | (16)–(26) | Vehicle coupling | Stage-location coupling ($X_i^k, A_h^i, B_h^i, Y_h^k, W_h^k$). |
| **Timing & Big-M** | (27)–(33) | Physical schedule | Synchronized arrival ($a_i$) and departure ($d_i$) times. |
| **Strengthening** | (34)–(40) | Valid inequalities | Includes Equation (34) and Equation (35). |
| **Domains & Tightening** | (41)–(55) | Bounds & integrality | Explicit impossible-variable fixings to zero. |

### Equation (35) Mathematical Proof

Equation (35) is:

$$\sum_{k=1}^K k X_E^k + \sum_{h \in C} \phi_h = N + 2$$

- In 1-based indexing, let $k_E$ be the stage where the truck arrives at the final depot $E$ ($X_E^{k_E} = 1$).
- Stage 1 is start depot $S$, and stage $k_E$ is end depot $E$.
- The intermediate stages $2, \dots, k_E - 1$ visit exactly $k_E - 2$ customers by truck.
- By Equation (20), every customer is served by either truck or drone:
  $$(\text{truck-served customers}) + (\text{drone-served customers}) = N$$
  $$(k_E - 2) + \sum_{h \in C} \phi_h = N \iff k_E + \sum_{h \in C} \phi_h = N + 2$$
- Therefore, the constant RHS is **always $N + 2$** (the total number of original customers plus 2 depots), regardless of the number of stages instantiated in subproblems.

---

## 2. Construct Step & Decoupling of Fixed Modes

The paper describes the Construct step as:
1. Sample a subset of truck customers ($C_{\text{truck}}$).
2. Solve MTZ TSP for the truck tour on $\{S\} \cup C_{\text{truck}} \cup \{E\}$.
3. Integrate remaining customers ($C_{\text{drone}}$) via the 2-index stage-based MILP with a fixed number of stages equal to the tour length plus 2 ($K = |C_{\text{truck}}| + 2$).

### Orthogonal Mathematical Modes

The codebase distinguishes four separate concepts:
1. **`num_stages`** ($K = |C_{\text{truck}}| + 2$): Sets the stage horizon for the subproblem.
2. **`fixed_truck_customers`** ($C_{\text{truck}}$): Forces $\phi_h = 0$ for $h \in C_{\text{truck}}$.
3. **`fixed_drone_customers`** ($C_{\text{drone}}$): Forces $\phi_h = 1$ and $X_h^k = 0$ for $h \in C_{\text{drone}}$.
4. **`fixed_truck_route`** (exact sequence): Forces $X_{\text{route}[k]}^k = 1$ and $x_{u, v} = 1$ for route arcs.

**Implementation hierarchy in Construct:**
- **Flexible Route Mode**: First solves with fixed stages $K = |C_{\text{truck}}| + 2$ and fixed customer sets without locking $x_{ij}$, allowing the truck to reorder customers if beneficial for drone coordination.
- **Fixed Route Mode**: If flexible solve exceeds half the sub-budget, falls back to locked truck sequence.
- **Heuristic Fallback**: If MILP cannot find a solution, attempts greedy consecutive-edge assignment.
- **All-Truck Fallback**: If drone assignment fails entirely, returns certified all-truck tour.

Every solution strictly records its actual generation method in `solution.status` and `solution.metadata["construction_method"]`:
`"stage_based_milp"`, `"heuristic_fallback"`, or `"all_truck_fallback"`.

---

## 3. CMSA Metaheuristic (Algorithm 1)

CMSA adheres strictly to Algorithm 1 of the paper:

```text
Algorithm 1 CMSA for FSTSP
1: age[c] = -1, ∀ c ∈ C (disable all components)
2: while time doesn't reach limit:
3:    s = Construct()
4:    age[c] = 0, ∀ c ∈ s (mark useful)
5:    Build restricted MILP with active components (age >= 0)
6:    sMIP = solve_restricted_milp(tMIP = 15s)  -- FULL stage horizon K = N + 2
7:    if sMIP is feasible:
8:        age[c] = 0, ∀ c ∈ sMIP (mark useful)
9:    for c ∈ C where age[c] != -1:
10:       age[c] += 1
11:       if age[c] == age_limit:
12:           age[c] = -1 (disable)
13:   Update sbest with min(s, sMIP)
14: return sbest
```

**Key guarantees:**
- **No protected set**: Every active component ages at each Adapt step without artificial immunity.
- **Full stage horizon in Solve**: The restricted MILP uses the full stage horizon ($K = N + 2$), never fixing stages to the Construct count.
- **Monotonicity**: `sbest` is updated evaluating both constructed and restricted solutions.

---

## 4. Table 4 Model Sizing Methodology

### CPLEX vs HiGHS Measurement Differences

Table 4 in the paper explicitly cites:
> *"Statistic information regarding the number of variables, constraints, nonzero coefficients reported by CPLEX"*

CPLEX applies an aggressive presolve routine that eliminates fixed variables ($UB = LB = 0$ or $1$) and prunes redundant constraint rows. Therefore, the paper reports **post-presolve dimensions**.

In SciPy / HiGHS, post-presolve dimensions are not exposed via the Python wrapper. To prevent methodological conflation, the repository reports separated metrics:

- `original_variables`: Total structural variables in the full formulation.
- `original_constraints`: Total rows in the constraint matrix.
- `original_nonzeros`: Total non-zeros in the constraint matrix.
- `fixed_zero_variables`: Count of variables with $LB = 0$ and $UB = 0$.
- `fixed_one_variables`: Count of variables with $LB = 1$ and $UB = 1$.
- `free_variables`: Count of active search variables ($UB > LB$).
- `active_constraints_before_presolve`: Constraint rows containing $\ge 1$ free variable.
- `active_nonzeros_before_presolve`: Non-zero entries on free variables.
- `presolved_variables`: `None` (annotated as unavailable from SciPy wrapper).

**Bug Fix Verification:**
Variables fixed to 1 (`fixed_one_variables`) are explicitly separated from variables fixed to 0 (`fixed_zero_variables`), eliminating variable-counting skew.

---

## 5. Experimental Protocol & Time Limits

| Experiment | Target Instances | Solver & Limits | Baseline Reference |
|---|---|---|---|
| **Table 1** | Agatz maxradius ($n=10, 20$) | Exact MILP, 1 hour (3600s) | Paper Table 1 |
| **Table 2** | Agatz novisit ($n=10$) | Exact MILP, 1 hour (3600s) | Paper Table 2 |
| **Table 3 Exact** | 40 synthetic instances ($n=20..50$) | CPLEX Exact, 2 hours (7200s) | Paper Table 3 |
| **Table 3 CMSA** | 40 synthetic instances ($n=20..50$) | CMSA ($t=1800\text{s}$, $t_{\text{MIP}}=15\text{s}$, $age=2$) | Paper Table 3 |
| **Table 4** | 40 synthetic instances ($n=20..50$) | CMSA model sizing ($age=2$ vs $age=5$) | Paper Table 4 |
| **Figure 1** | $n=30$ seed 1 | Real Exact route vs CMSA route | Paper Figure 1 |

### Data Transparency

The 40 benchmark instances in this repository are an independent synthetic reproduction generated strictly according to Agatz et al.'s protocol ($n \in \{20, 30, 40, 50\}$, uniform $[0, 100]^2$, depot $(50, 50)$, maxradius = 200%, truck speed 1, drone speed 2) because the authors did not publish raw coordinate files or seeds. Numerical comparisons reflect statistical cross-instance evaluation.
