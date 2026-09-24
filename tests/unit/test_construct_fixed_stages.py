from __future__ import annotations

import numpy as np
import pytest

from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution


def test_construct_fixed_stages_and_validity():
    """Verify construct_solution builds a valid solution using fixed number of stages."""
    inst = generate_uniform_instance(n=8, seed=42)
    rng = np.random.default_rng(42)

    sol = construct_solution(
        inst,
        rng,
        truck_sample_ratio=0.6,
        solver_backend="highs",
        exact_tsp_threshold=60,
    )

    assert sol.feasible
    issues = validate_solution(inst, sol)
    assert issues == []

    # Check that truck route starts at start depot and ends at end depot
    assert sol.truck_route[0] == 0
    assert sol.truck_route[-1] == inst.n + 1

    # Check coverage of all customers: truck customers + drone customers = n
    truck_custs = set(sol.truck_route[1:-1])
    drone_custs = set(s.customer for s in sol.drone_sorties)
    assert len(truck_custs.intersection(drone_custs)) == 0
    assert truck_custs.union(drone_custs) == set(range(1, inst.n + 1))
