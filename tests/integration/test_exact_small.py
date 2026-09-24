from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model


def test_exact_small_is_feasible():
    inst = generate_uniform_instance(3, seed=1, width=20, drone_endurance=100)
    sol = solve_stage_model(inst, time_limit=10, mip_rel_gap=0.0)
    assert sol.feasible
    assert sol.objective is not None
    assert not validate_solution(inst, sol)
