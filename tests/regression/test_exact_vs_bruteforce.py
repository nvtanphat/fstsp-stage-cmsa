import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.bruteforce import brute_force_optimum
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model


@pytest.mark.parametrize("n,seed", [(2, 1), (2, 4), (3, 2), (3, 6), (4, 1), (4, 5)])
def test_stage_milp_matches_independent_bruteforce(n: int, seed: int):
    inst = generate_uniform_instance(
        n,
        seed=seed,
        width=30.0,
        launch_time=0.0,
        recovery_time=0.0,
    )
    oracle = brute_force_optimum(inst, max_customers=4)
    exact = solve_stage_model(inst, time_limit=30.0, mip_rel_gap=0.0)

    assert oracle.feasible
    assert exact.feasible
    assert validate_solution(inst, exact) == []
    assert exact.objective == pytest.approx(oracle.objective, abs=2e-5)
