from __future__ import annotations

import pytest

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.data.generator import generate_uniform_instance


def test_table4_presolve_metric_integrity():
    """Verify that free variables and active components are never substituted for presolved metrics."""
    inst = generate_uniform_instance(n=6, seed=1)
    sol = solve_cmsa(inst, total_time=3.0, mip_time=1.0, age_limit=2, seed=1, solver_backend="highs")
    assert sol.feasible

    hist = sol.metadata.get("history", [])
    assert len(hist) > 0

    for it in hist:
        # 1. Under HiGHS, presolved metrics must be None, NOT substituted by free_variables!
        assert it.get("presolved_variables") is None
        assert it.get("presolved_constraints") is None
        assert it.get("presolved_nonzeros") is None

        # 2. active_components (number of graph edges/sorties) is distinct from mathematical variables
        if it.get("active_components") is not None and it.get("original_variables") is not None:
            assert isinstance(it["active_components"], int)
            assert it["original_variables"] > 0
            # active components is small set size, not the full MILP variable dimension
            assert it["active_components"] <= it["original_variables"]

    # Verify at least one iteration successfully ran restricted MIP and recorded metrics
    valid_iters = [it for it in hist if it.get("original_variables") is not None]
    assert len(valid_iters) > 0
