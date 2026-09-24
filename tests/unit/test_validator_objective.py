import math

from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.solution import FSTSPSolution
from fstsp.evaluation.schedule import evaluate_schedule
from fstsp.evaluation.validator import validate_solution


def _all_truck_solution(inst):
    route = [inst.S, *inst.customers, inst.E]
    provisional = FSTSPSolution(feasible=True, objective=0.0, truck_route=route)
    ev = evaluate_schedule(inst, provisional)
    assert ev.feasible and ev.completion_time is not None
    provisional.objective = ev.completion_time
    return provisional


def test_validator_rejects_missing_objective_on_feasible_solution():
    inst = generate_uniform_instance(2, seed=7)
    sol = _all_truck_solution(inst)
    sol.objective = None
    issues = validate_solution(inst, sol)
    assert any("missing an objective" in x for x in issues)


def test_validator_rejects_nonfinite_objective():
    inst = generate_uniform_instance(2, seed=8)
    sol = _all_truck_solution(inst)
    sol.objective = math.inf
    issues = validate_solution(inst, sol)
    assert any("not finite" in x for x in issues)
