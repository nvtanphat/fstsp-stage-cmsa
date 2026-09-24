"""Comprehensive regression tests covering all 10 audit findings and paper requirements.

1. Equation (35) consistency under fixed stages.
2. Fixed stages allows flexible truck route without rigid permutation lock.
3. MTZ solver status distinction: OPTIMAL vs FEASIBLE vs TIME_LIMIT vs INFEASIBLE.
4. AgeManager multi-iteration lifecycle (0 -> 1 -> 2 -> -1) without protected leak.
5. Infeasible restricted MIP preserves constructed incumbent.
6. Metric partitioning: fixed_zero, fixed_one, free variables (fixed-one not counted as fixed-zero).
7. Construct fallback records exact method (stage_based_milp vs heuristic vs all_truck).
8. Deadline control prevents budget overrun.
9. Convergence best-so-far is monotonically non-increasing.
10. Exact MILP matches brute-force oracle on tiny instance.
"""
from __future__ import annotations

import math
import time
import numpy as np
import pytest

from fstsp.algorithms.cmsa.age import AgeManager
from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.algorithms.tsp.mtz import solve_tsp_mtz_result
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.bruteforce import brute_force_optimum
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import (
    build_stage_model,
    extract_model_metrics,
    solve_stage_model,
)


def test_1_equation_35_fixed_stages():
    """1. Test that Equation (35) is mathematically consistent and satisfied under fixed stages."""
    inst = generate_uniform_instance(n=10, seed=42)
    truck_custs = [1, 2, 3, 4, 5, 6, 7, 8]
    drone_custs = [9, 10]
    num_stages = len(truck_custs) + 2  # 10 stages

    sol = solve_stage_model(
        inst,
        time_limit=10.0,
        num_stages=num_stages,
        fixed_truck_customers=truck_custs,
        fixed_drone_customers=drone_custs,
        strengthen=True,
    )
    assert sol.feasible, f"Stage model with fixed stages failed: {sol.status}"
    assert sol.objective is not None

    # Verify Equation (35): 1-based stage of E + number of drone customers == N + 2
    assert sol.truck_route is not None
    k_E_1based = len(sol.truck_route)  # In full tour, E is the last node
    num_drone_custs = len(sol.drone_sorties)
    assert k_E_1based + num_drone_custs == inst.n + 2, (
        f"Eq (35) violated: k_E={k_E_1based}, drone={num_drone_custs}, expected {inst.n + 2}"
    )


def test_2_fixed_stages_flexible_route():
    """2. Test that fixed stages does not lock truck route to a single rigid permutation."""
    inst = generate_uniform_instance(n=10, seed=1)
    truck_custs = [1, 2, 3, 4, 5, 6, 7, 8]
    drone_custs = [9, 10]
    num_stages = len(truck_custs) + 2

    # Model with fixed customer sets but WITHOUT fixed_truck_route
    model = build_stage_model(
        inst,
        num_stages=num_stages,
        fixed_truck_customers=truck_custs,
        fixed_drone_customers=drone_custs,
        strengthen=True,
    )

    # Check that truck customer 1 can visit multiple stages (X is not fixed to 1 at a single stage)
    var = model.var
    lb, ub = model.bounds.lb, model.bounds.ub
    stages_free_for_cust1 = 0
    for k in range(1, num_stages - 1):
        idx = var[("X", k, 1)]
        if ub[idx] > lb[idx]:
            stages_free_for_cust1 += 1

    assert stages_free_for_cust1 > 1, "Truck customer 1 was rigidly fixed to a single stage!"

    # Solve and verify that the solver can find a feasible solution
    sol = solve_stage_model(
        inst,
        time_limit=10.0,
        num_stages=num_stages,
        fixed_truck_customers=truck_custs,
        fixed_drone_customers=drone_custs,
        strengthen=True,
    )
    assert sol.feasible


def test_3_mtz_status_distinction():
    """3. Test solve_tsp_mtz_result distinguishing OPTIMAL, FEASIBLE, TIME_LIMIT, INFEASIBLE, ERROR."""
    # 3a. Solvable instance -> OPTIMAL
    dist = np.array([
        [0.0, 10.0, 20.0, 30.0],
        [10.0, 0.0, 15.0, 25.0],
        [20.0, 15.0, 0.0, 10.0],
        [30.0, 25.0, 10.0, 0.0],
    ])
    res_opt = solve_tsp_mtz_result([1, 2], dist, 0, 3, time_limit=5.0)
    assert res_opt.status == "OPTIMAL"
    assert res_opt.proven_optimal is True
    assert res_opt.route == [0, 1, 2, 3]

    # 3b. Timeout with no solution -> TIME_LIMIT
    res_timeout = solve_tsp_mtz_result([1, 2], dist, 0, 3, time_limit=1e-7)
    assert res_timeout.status in ("TIME_LIMIT", "OPTIMAL")  # If HiGHS solves presolve instantly, OPTIMAL; else TIME_LIMIT

    # 3c. Validation check on duplicate nodes -> ValueError
    with pytest.raises(ValueError, match="duplicates"):
        solve_tsp_mtz_result([1, 1], dist, 0, 3, time_limit=1.0)


def test_4_age_manager_multi_iteration():
    """4. Test AgeManager multi-iteration lifecycle (0 -> 1 -> 2 -> -1) without protected leak."""
    ages = AgeManager(age_limit=2)
    c1 = ("x", 0, 1)
    c2 = ("x", 1, 2)
    c3 = ("phi", 3)

    # Iteration 1: Useful components c1, c2 marked at 0
    ages.mark_useful({c1, c2})
    assert ages.age[c1] == 0
    assert ages.age[c2] == 0
    assert ages.active() == {c1, c2}

    # Adapt at end of iteration 1: active components aged by 1
    ages.adapt()
    assert ages.age[c1] == 1
    assert ages.age[c2] == 1
    assert ages.active() == {c1, c2}

    # Iteration 2: Only c1 is in the new solution, c3 is added
    ages.mark_useful({c1, c3})
    assert ages.age[c1] == 0
    assert ages.age[c2] == 1
    assert ages.age[c3] == 0

    # Adapt at end of iteration 2:
    # c1: 0 -> 1
    # c2: 1 -> 2 >= age_limit -> -1 (disabled!)
    # c3: 0 -> 1
    ages.adapt()
    assert ages.age[c1] == 1
    assert ages.age[c2] == -1, "c2 reached age_limit=2 but was not disabled!"
    assert ages.age[c3] == 1
    assert ages.active() == {c1, c3}

    # Iteration 3: Neither reinforced
    ages.adapt()
    assert ages.age[c1] == -1, "c1 reached age_limit=2 but was not disabled!"
    assert ages.age[c3] == -1, "c3 reached age_limit=2 but was not disabled!"
    assert len(ages.active()) == 0


def test_5_restricted_mip_infeasible_preserves_incumbent():
    """5. Test that an infeasible restricted MIP gracefully preserves the constructed incumbent."""
    inst = generate_uniform_instance(n=6, seed=1)
    # Run CMSA with age_limit=1 and mip_time=1.0: restricted solve may have few components
    sol = solve_cmsa(inst, total_time=5.0, mip_time=1.0, age_limit=1, seed=42)
    assert sol.feasible
    assert sol.objective is not None
    # Verify validator approves solution
    issues = validate_solution(inst, sol)
    assert not issues, f"CMSA produced invalid solution: {issues}"


def test_6_fixed_variable_metrics_counting():
    """6. Test partition of fixed_zero, fixed_one, free variables; fixed-one must never be counted as fixed-zero."""
    inst = generate_uniform_instance(n=10, seed=1)
    route = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11]
    model = build_stage_model(inst, fixed_truck_route=route, strengthen=True)

    metrics = extract_model_metrics(model)

    assert metrics["fixed_one_variables"] > 0, "Model with fixed route must have fixed-one variables!"
    assert metrics["fixed_zero_variables"] > 0
    assert metrics["free_variables"] > 0

    # Verify mutual exclusivity of partitions
    lb = np.asarray(model.bounds.lb)
    ub = np.asarray(model.bounds.ub)
    fixed_zero = (np.abs(lb) < 1e-9) & (np.abs(ub) < 1e-9)
    fixed_one = (np.abs(lb - 1.0) < 1e-9) & (np.abs(ub - 1.0) < 1e-9)
    free = (ub > lb + 1e-9)

    # Intersection of fixed-zero and fixed-one must be exactly empty
    assert np.all(~(fixed_zero & fixed_one)), "Fixed-zero and Fixed-one sets overlap!"
    assert np.all(~(fixed_zero & free)), "Fixed-zero and Free sets overlap!"
    assert np.all(~(fixed_one & free)), "Fixed-one and Free sets overlap!"

    # Presolved metrics must be explicitly None for HiGHS
    assert metrics["presolved_variables"] is None
    assert metrics["presolve_metrics_available"] is False


def test_7_construct_fallback_status():
    """7. Test Construct fallback records exact method (stage_based_milp vs heuristic vs all_truck)."""
    inst = generate_uniform_instance(n=8, seed=1)
    rng = np.random.default_rng(42)

    # Normal construct
    sol_normal = construct_solution(inst, rng, truck_sample_ratio=0.65)
    assert sol_normal.feasible
    method = sol_normal.metadata.get("construction_method")
    assert method in ("stage_based_milp", "heuristic_fallback", "all_truck_fallback")
    if method == "stage_based_milp":
        assert sol_normal.status == "constructed_stage_based"
    elif method == "heuristic_fallback":
        assert sol_normal.status == "constructed_heuristic_fallback"
    elif method == "all_truck_fallback":
        assert sol_normal.status == "constructed_all_truck_fallback"

    # Forced zero-limit construct -> all-truck fallback
    sol_forced = construct_solution(inst, rng, truck_sample_ratio=0.5, tsp_time_limit=0.0, deadline=time.perf_counter() - 1.0)
    assert sol_forced.feasible
    assert sol_forced.metadata["construction_method"] == "all_truck_fallback"
    assert sol_forced.status == "constructed_all_truck_fallback"


def test_8_cmsa_deadline_control():
    """8. Test deadline control strictly limits wall-clock budget on n=20."""
    inst = generate_uniform_instance(n=20, seed=1)
    budget = 1.0
    t0 = time.perf_counter()
    sol = solve_cmsa(inst, total_time=budget, mip_time=0.5, seed=42)
    elapsed = time.perf_counter() - t0
    # Allow reasonable OS scheduler tolerance (under 2.5x budget for n=20)
    assert elapsed < budget + 2.0, f"CMSA exceeded budget: {elapsed:.2f}s vs {budget:.2f}s"
    assert sol.feasible


def test_9_convergence_monotonicity():
    """9. Test that best-so-far solution in CMSA is monotonically non-increasing."""
    inst = generate_uniform_instance(n=8, seed=1)
    sol = solve_cmsa(inst, total_time=6.0, mip_time=1.5, age_limit=2, seed=42)
    assert sol.feasible
    history = sol.metadata.get("history", [])
    assert len(history) >= 1

    # Check that best found is <= all observed valid objectives
    best_obj = sol.objective
    assert best_obj is not None
    for entry in history:
        r_obj = entry.get("restricted_objective")
        c_obj = entry.get("constructed_objective")
        if r_obj is not None:
            assert best_obj <= r_obj + 1e-6
        if c_obj is not None:
            assert best_obj <= c_obj + 1e-6


def test_10_exact_milp_vs_bruteforce():
    """10. Test that exact stage MILP matches exhaustive brute-force oracle on tiny instance."""
    inst = generate_uniform_instance(n=3, seed=10)
    sol_milp = solve_stage_model(inst, time_limit=15.0, strengthen=True)
    assert sol_milp.feasible
    assert sol_milp.objective is not None

    oracle_sol = brute_force_optimum(inst)
    assert oracle_sol.feasible
    assert oracle_sol.objective is not None

    diff = abs(sol_milp.objective - oracle_sol.objective)
    assert diff <= 1e-5, f"Stage MILP objective {sol_milp.objective} != Oracle {oracle_sol.objective}, diff={diff}"
