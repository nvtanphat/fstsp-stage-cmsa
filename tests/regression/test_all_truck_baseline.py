from fstsp.data.generator import generate_uniform_instance
from fstsp.formulation.stage_based import solve_stage_model


def test_objective_positive():
    inst = generate_uniform_instance(2, seed=9, width=10, drone_endurance=100)
    sol = solve_stage_model(inst, time_limit=10, mip_rel_gap=0.0)
    assert sol.feasible
    assert sol.objective > 0
