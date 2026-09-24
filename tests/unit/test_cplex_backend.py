from __future__ import annotations

import pytest

from fstsp.solver.cplex_backend import CplexBackend, CplexNotAvailableError
from fstsp.solver.base import SolverOptions
from fstsp.formulation.stage_based import build_stage_model
from fstsp.data.generator import generate_uniform_instance


def test_cplex_backend_class_properties():
    backend = CplexBackend()
    assert hasattr(backend, "solve")
    assert hasattr(backend, "is_available")


def test_cplex_backend_solve_raises_when_unavailable():
    if CplexBackend.is_available():
        pytest.skip("CPLEX is installed; testing unavailability is skipped.")

    inst = generate_uniform_instance(4, seed=42)
    model = build_stage_model(inst)
    backend = CplexBackend()
    options = SolverOptions(time_limit=10.0)

    with pytest.raises(CplexNotAvailableError) as exc_info:
        backend.solve(model, options)

    assert "CPLEX backend was requested per paper protocol" in str(exc_info.value)


@pytest.mark.skipif(not CplexBackend.is_available(), reason="Requires IBM ILOG CPLEX license")
def test_cplex_backend_real_solve():
    """Integration test that executes ONLY when real IBM ILOG CPLEX is installed.

    Verifies that variables, objective, constraints, and presolve metrics are queried directly from CPLEX.
    """
    inst = generate_uniform_instance(4, seed=42)
    model = build_stage_model(inst)
    backend = CplexBackend()
    options = SolverOptions(time_limit=15.0, threads=2, mip_emphasis=5)

    result = backend.solve(model, options)

    assert result.solver_name == "cplex"
    assert result.solver_version is not None
    assert result.success is True
    assert result.x is not None
    assert result.objective is not None
    assert result.node_count is not None
    assert "solver_reported_variables" in result.model_metrics
    assert result.model_metrics["solver_reported_variables"] == model.var.size
