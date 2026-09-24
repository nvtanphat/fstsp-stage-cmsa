import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.formulation.stage_based import solve_stage_model


@pytest.mark.parametrize("n,seed", [(2, 1), (3, 2), (4, 3)])
def test_strengthening_does_not_change_tiny_optimum(n, seed):
    inst = generate_uniform_instance(
        n,
        seed=seed,
        width=25.0,
        drone_endurance=60.0,
        launch_time=1.0,
        recovery_time=1.0,
        novisit_fraction=0.25,
    )
    strong = solve_stage_model(inst, time_limit=10.0, mip_rel_gap=0.0, strengthen=True)
    plain = solve_stage_model(inst, time_limit=10.0, mip_rel_gap=0.0, strengthen=False)
    assert strong.feasible and plain.feasible
    assert strong.objective == pytest.approx(plain.objective, abs=1e-5)
