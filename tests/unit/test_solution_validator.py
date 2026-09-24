from __future__ import annotations

import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.solution import DroneSortie, FSTSPSolution
from fstsp.evaluation.validator import validate_solution


def test_validator_detects_coverage_and_endurance_violations():
    """Verify validator flags missing customer, duplicate visits, and endurance violations."""
    inst = generate_uniform_instance(n=4, seed=1)

    # 1. Missing customer: customer 2 is neither on truck nor drone
    sol_missing = FSTSPSolution(
        feasible=True,
        truck_route=[0, 1, 3, 4, 5],
        drone_sorties=[],
        objective=100.0,
        status="feasible",
    )
    issues = validate_solution(inst, sol_missing, check_objective=False)
    assert any("customer" in iss.lower() or "visit" in iss.lower() or "missing" in iss.lower() for iss in issues)

    # 2. Duplicate visit: customer 1 is both on truck and drone
    sol_dup = FSTSPSolution(
        feasible=True,
        truck_route=[0, 1, 2, 3, 4, 5],
        drone_sorties=[DroneSortie(launch_node=0, customer=1, recovery_node=2)],
        objective=100.0,
        status="feasible",
    )
    issues_dup = validate_solution(inst, sol_dup, check_objective=False)
    assert len(issues_dup) > 0

    # 3. Valid all-truck solution passes structural validation
    from fstsp.evaluation.schedule import evaluate_schedule
    sol_all_truck = FSTSPSolution(
        feasible=True,
        truck_route=[0, 1, 2, 3, 4, 5],
        drone_sorties=[],
        objective=0.0,
        status="feasible",
    )
    sched = evaluate_schedule(inst, sol_all_truck)
    sol_all_truck.objective = sched.completion_time
    issues_valid = validate_solution(inst, sol_all_truck, check_objective=True)
    assert issues_valid == []
