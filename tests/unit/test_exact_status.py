from __future__ import annotations

import pytest

from fstsp.domain.solution import FSTSPSolution
import kaggle.run_experiment as kre


def test_map_exact_status_optimal():
    sol = FSTSPSolution(
        feasible=True,
        objective=100.0,
        runtime=12.5,
        status="optimal",
        metadata={"proven_optimal": True},
    )
    status, timeout, proven = kre.map_exact_status(sol, time_limit=60.0)
    assert status == "OPTIMAL"
    assert timeout is False
    assert proven is True


def test_map_exact_status_time_limit():
    sol = FSTSPSolution(
        feasible=True,
        objective=120.0,
        runtime=60.0,
        status="time_limit_exhausted",
        metadata={"proven_optimal": False},
    )
    status, timeout, proven = kre.map_exact_status(sol, time_limit=60.0)
    assert status == "TIME_LIMIT"
    assert timeout is True
    assert proven is False


def test_map_exact_status_infeasible():
    sol = FSTSPSolution(
        feasible=False,
        objective=float("inf"),
        runtime=2.0,
        status="infeasible",
        metadata={"proven_optimal": False},
    )
    status, timeout, proven = kre.map_exact_status(sol, time_limit=60.0)
    assert status == "INFEASIBLE"
    assert timeout is False
    assert proven is False


def test_skipped_exact_run_status():
    """Verify that unexecuted exact runs have status NOT_RUN and timeout False."""
    res_exact = {
        "tag": "table3_n40_seed1_exact",
        "method": "exact",
        "feasible": False,
        "objective": None,
        "runtime": 0.0,
        "status": "NOT_RUN",
        "timeout": False,
        "proven_optimal": False,
    }
    assert res_exact["status"] == "NOT_RUN"
    assert res_exact["timeout"] is False
