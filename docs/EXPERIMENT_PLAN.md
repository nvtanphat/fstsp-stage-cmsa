# Experiment plan

## Phase A — formulation verification

Run n=2..8 with exact MILP. Inspect customer coverage, objective, route plots, and solver status. Keep seeds fixed.

## Phase B — CMSA smoke benchmark

Run n=10,20 with 3-10 seeds. Record objective, runtime, drone-customer count, active component count, and iteration history.

## Phase C — paper-scale direction

Run n=20,30,40,50 with `age_limit=2` and `mip_time=15`. Start with a short overall budget, then increase toward 30 minutes if Kaggle quota/runtime permits.

## Phase D — public restricted data

Use downloaded `maxradius` and `novisit` instances. Compare feasibility and objective trends; do not equate HiGHS runtime to CPLEX runtime.

## Reporting

Always report: solver, hardware/cloud environment, time limit, seed, n, objective, feasibility, MIP gap if available, and implementation deviations.
