import numpy as np
import pytest

from fstsp.domain.instance import FSTSPInstance
from fstsp.evaluation.bruteforce import brute_force_optimum
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model


@pytest.mark.parametrize("seed", list(range(12)))
def test_randomized_small_instances_match_independent_oracle(seed: int):
    rng = np.random.default_rng(9000 + seed)
    n = 3 if seed < 8 else 4
    inst = FSTSPInstance(
        name=f"fuzz_{seed}",
        coords=rng.uniform(-20.0, 20.0, size=(n, 2)),
        depot_coord=np.array([0.0, 0.0]),
        truck_speed=float(rng.uniform(0.6, 1.8)),
        drone_speed=float(rng.uniform(1.0, 3.5)),
        drone_endurance=float(rng.uniform(8.0, 70.0)),
        launch_time=float(rng.choice([0.0, 0.5, 1.0, 2.0])),
        recovery_time=float(rng.choice([0.0, 0.5, 1.0, 2.0])),
        drone_allowed=(rng.random(n) > 0.25),
    )
    oracle = brute_force_optimum(inst, max_customers=4)
    exact = solve_stage_model(inst, time_limit=20.0, mip_rel_gap=0.0)
    assert oracle.feasible and exact.feasible
    assert validate_solution(inst, exact) == []
    assert exact.objective == pytest.approx(oracle.objective, abs=2e-5)
    assert exact.metadata["max_row_violation"] <= exact.metadata["vector_validation_tolerance"]
    assert exact.metadata["max_integrality_violation"] <= exact.metadata["vector_validation_tolerance"]
