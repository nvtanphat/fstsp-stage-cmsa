import pytest
import numpy as np

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution


def test_construct_can_skip_exact_tsp_when_budget_is_zero():
    inst = generate_uniform_instance(8, seed=9)
    sol = construct_solution(inst, np.random.default_rng(9), tsp_time_limit=0.0)
    assert sol.feasible
    assert validate_solution(inst, sol) == []


def test_cmsa_rejects_invalid_gap():
    inst = generate_uniform_instance(3, seed=1)
    with pytest.raises(ValueError):
        solve_cmsa(inst, total_time=1.0, mip_time=0.1, mip_rel_gap=-0.1)
