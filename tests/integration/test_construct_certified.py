import numpy as np
import pytest

from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution


@pytest.mark.parametrize("n", [8, 10, 12])
def test_construct_is_certified_across_hard_random_cases(n: int):
    # Low endurance + NOVISIT used to trigger stale sortie stages after route edits.
    for seed in range(20):
        inst = generate_uniform_instance(
            n,
            seed=seed,
            width=100.0,
            drone_endurance=30.0,
            launch_time=1.0,
            recovery_time=1.0,
            novisit_fraction=0.2,
        )
        sol = construct_solution(
            inst,
            np.random.default_rng(seed),
            truck_sample_ratio=0.5,
            exact_tsp_threshold=8,
        )
        assert sol.feasible
        assert validate_solution(inst, sol) == []
