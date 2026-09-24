from __future__ import annotations

import pytest

from fstsp.solver import (
    CplexBackend,
    CplexNotAvailableError,
    HighsBackend,
    get_solver_backend,
)
from fstsp.data.generator import generate_uniform_instance
from fstsp.formulation.stage_based import solve_stage_model


def test_get_solver_backend_highs():
    backend = get_solver_backend("highs")
    assert isinstance(backend, HighsBackend)
    assert backend.is_available() is True

    backend_scipy = get_solver_backend("scipy")
    assert isinstance(backend_scipy, HighsBackend)


def test_get_solver_backend_cplex():
    backend = get_solver_backend("cplex")
    assert isinstance(backend, CplexBackend)


def test_get_solver_backend_unsupported():
    with pytest.raises(ValueError, match="Unsupported solver backend"):
        get_solver_backend("gurobi_unknown")


def test_cplex_backend_does_not_fall_back_silently():
    """Verify that when CPLEX is selected on a system without CPLEX,

    it strictly raises CplexNotAvailableError and does NOT silently fall back to HiGHS.
    """
    if CplexBackend.is_available():
        pytest.skip("CPLEX is installed on this system; testing absence behavior is not applicable.")

    inst = generate_uniform_instance(4, seed=42)

    with pytest.raises(CplexNotAvailableError) as exc_info:
        solve_stage_model(inst, solver_backend="cplex")

    msg = str(exc_info.value).lower()
    assert "cplex backend was requested" in msg
    assert "not silently fall back to highs" in msg
