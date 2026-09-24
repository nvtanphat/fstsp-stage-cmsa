from __future__ import annotations

import pandas as pd
import pytest

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.data.generator import generate_uniform_instance


def test_table4_iteration_collection():
    """Verify that restricted MIP iterations separate pre-presolve and presolved dimensions."""
    inst = generate_uniform_instance(n=6, seed=1)
    sol = solve_cmsa(inst, total_time=3.0, mip_time=1.0, age_limit=2, seed=1, solver_backend="highs")
    assert sol.feasible
    hist = sol.metadata.get("history", [])
    assert len(hist) > 0

    first_it = hist[0]
    # Check that structural metrics exist
    assert "original_variables" in first_it or "n_variables" in first_it
    assert "free_variables" in first_it or "n_active_variables" in first_it
    assert "active_constraints_before_presolve" in first_it or "n_active_constraints" in first_it
    assert "active_nonzeros_before_presolve" in first_it or "n_active_nonzeros" in first_it

    # Check presolved metrics: under HiGHS, presolved_* must be None, NOT substituted by free_variables!
    assert first_it.get("presolved_variables") is None
    assert first_it.get("presolved_constraints") is None
    assert first_it.get("presolved_nonzeros") is None

    # Check active_components is strictly separate
    if "active_components" in first_it:
        assert isinstance(first_it["active_components"], int)
        # active_components is number of graph edges/sorties, not MILP variable count
        n_orig = first_it.get("original_variables", first_it.get("n_variables"))
        assert first_it["active_components"] <= n_orig
