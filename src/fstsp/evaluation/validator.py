from __future__ import annotations

from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import FSTSPSolution
from fstsp.evaluation.schedule import evaluate_schedule


def validate_solution(
    instance: FSTSPInstance,
    solution: FSTSPSolution,
    *,
    check_objective: bool = True,
    objective_atol: float = 1e-5,
    objective_rtol: float = 1e-7,
) -> list[str]:
    """Validate structural, synchronization, endurance and objective consistency.

    This is intentionally independent of the MILP solver. A result is only treated
    as research-valid when the discrete truck/drone plan can be reconstructed into
    a feasible physical schedule.
    """
    if not solution.feasible:
        return ["solution marked infeasible"]

    evaluation = evaluate_schedule(instance, solution)
    issues = list(evaluation.issues)
    if issues or not check_objective:
        return issues

    if solution.objective is None:
        issues.append("feasible solution is missing an objective value")
        return issues

    try:
        reported = float(solution.objective)
    except (TypeError, ValueError):
        issues.append("reported objective is not numeric")
        return issues

    import math
    if not math.isfinite(reported):
        issues.append("reported objective is not finite")
        return issues

    assert evaluation.completion_time is not None
    diff = abs(reported - evaluation.completion_time)
    tol = objective_atol + objective_rtol * max(
        1.0, abs(reported), abs(evaluation.completion_time)
    )
    if diff > tol:
        issues.append(
            "objective mismatch: "
            f"reported={solution.objective:.10g}, reconstructed={evaluation.completion_time:.10g}"
        )
    return issues
