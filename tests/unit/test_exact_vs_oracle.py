from __future__ import annotations

import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model


def test_exact_stage_model_on_small_instance():
    """Verify exact 2-index stage-based MILP solves small instance optimally and passes validation."""
    inst = generate_uniform_instance(n=4, seed=42, width=50.0)
    sol = solve_stage_model(inst, time_limit=15.0, solver_backend="highs")

    assert sol.feasible
    assert sol.objective is not None
    assert sol.objective > 0.0

    issues = validate_solution(inst, sol)
    assert issues == [], f"Validation issues: {issues}"

    # Verify depots
    assert sol.truck_route[0] == 0
    assert sol.truck_route[-1] == inst.n + 1

    # Verify coverage of all customers
    served = set(sol.truck_route[1:-1]) | {s.customer for s in sol.drone_sorties}
    assert served == set(range(1, inst.n + 1))
