import numpy as np
from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution


def test_construct_covers_customers():
    inst = generate_uniform_instance(8, seed=3, drone_endurance=200)
    sol = construct_solution(inst, np.random.default_rng(3), truck_sample_ratio=0.7)
    assert sol.feasible
    assert not validate_solution(inst, sol)
