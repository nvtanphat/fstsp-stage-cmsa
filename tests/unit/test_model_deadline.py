import time

from fstsp.data.generator import generate_uniform_instance
from fstsp.formulation.stage_based import solve_stage_model


def test_stage_model_respects_expired_deadline_without_entering_solver():
    inst = generate_uniform_instance(8, seed=1)
    sol = solve_stage_model(
        inst,
        time_limit=30.0,
        deadline=time.perf_counter() - 1.0,
    )
    assert not sol.feasible
    assert sol.status == "model_build_deadline_exceeded"
