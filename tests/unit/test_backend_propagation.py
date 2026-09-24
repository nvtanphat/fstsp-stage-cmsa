from __future__ import annotations

import numpy as np
import pytest

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.data.generator import generate_uniform_instance


def test_construct_backend_metadata():
    """Verify construct_solution propagates and records solver backends."""
    inst = generate_uniform_instance(n=6, seed=1)
    rng = np.random.default_rng(42)

    sol = construct_solution(inst, rng, solver_backend="highs", exact_tsp_threshold=60)
    assert sol.feasible
    assert sol.metadata.get("solver_backend") == "highs"
    assert "construction_tsp_backend" in sol.metadata
    assert "construction_integration_backend" in sol.metadata
    assert sol.metadata["construction_tsp_backend"] in ("highs", "heuristic_2opt")
    assert sol.metadata["construction_integration_backend"] in ("highs", "heuristic_greedy")


def test_cmsa_backend_propagation():
    """Verify solve_cmsa propagates solver_backend to all phases and records in history and metadata."""
    inst = generate_uniform_instance(n=6, seed=1)
    sol = solve_cmsa(
        inst,
        total_time=4.0,
        mip_time=1.0,
        age_limit=2,
        seed=1,
        solver_backend="highs",
        threads=1,
    )
    assert sol.feasible
    assert sol.metadata.get("restricted_mip_backend") == "highs"
    assert "construction_tsp_backend" in sol.metadata
    assert "construction_integration_backend" in sol.metadata

    history = sol.metadata.get("history", [])
    assert len(history) > 0
    for it in history:
        assert it.get("restricted_mip_backend") == "highs"
        assert "construction_tsp_backend" in it
        assert "construction_integration_backend" in it
