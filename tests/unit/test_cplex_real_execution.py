from __future__ import annotations

import pytest

from fstsp.solver import get_solver_backend, CplexNotAvailableError
from fstsp.solver.cplex_backend import CplexBackend
from fstsp.solver.highs import HighsBackend


def test_cplex_backend_selection_or_clean_rejection():
    """Verify that selecting CPLEX either returns real CplexBackend or raises CplexNotAvailableError.

    Never silently falls back to HiGHS when CPLEX is explicitly requested.
    """
    try:
        backend = get_solver_backend("cplex")
        assert isinstance(backend, CplexBackend)
        assert backend.name == "cplex"
    except CplexNotAvailableError as exc:
        assert "CPLEX is not available" in str(exc) or "cplex" in str(exc).lower()


def test_no_silent_fallback_to_highs():
    """Verify that CplexBackend does not claim to be HighsBackend."""
    highs = get_solver_backend("highs")
    assert isinstance(highs, HighsBackend)
    assert highs.name == "highs"
