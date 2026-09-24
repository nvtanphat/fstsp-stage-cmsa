from __future__ import annotations

from fstsp.domain.solution import FSTSPSolution


def solution_metrics(solution: FSTSPSolution) -> dict:
    return {
        "feasible": solution.feasible,
        "objective": solution.objective,
        "runtime": solution.runtime,
        "mip_gap": solution.mip_gap,
        "truck_stops": max(0, len(solution.truck_route) - 2),
        "drone_customers": len(solution.drone_sorties),
        "status": solution.status,
    }
