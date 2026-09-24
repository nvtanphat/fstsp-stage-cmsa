from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.solution import DroneSortie, FSTSPSolution
from fstsp.evaluation.validator import validate_solution


def test_validator_rejects_stale_stage_metadata():
    inst = generate_uniform_instance(2, seed=1, launch_time=0.0, recovery_time=0.0)
    sol = FSTSPSolution(
        feasible=True,
        objective=999.0,
        truck_route=[inst.S, 1, inst.E],
        drone_sorties=[DroneSortie(inst.S, 2, inst.E, 1, 2)],
    )
    issues = validate_solution(inst, sol)
    assert any("launch_stage" in x for x in issues)


def test_validator_rejects_positive_launch_time_from_start_depot():
    inst = generate_uniform_instance(2, seed=1, launch_time=1.0, recovery_time=1.0)
    sol = FSTSPSolution(
        feasible=True,
        objective=1.0,
        truck_route=[inst.S, 1, inst.E],
        drone_sorties=[DroneSortie(inst.S, 2, inst.E, 0, 2)],
    )
    issues = validate_solution(inst, sol)
    assert any("positive launch_time" in x for x in issues)
