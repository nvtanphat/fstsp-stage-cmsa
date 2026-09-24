from __future__ import annotations

import time
import numpy as np

from fstsp.algorithms.cmsa.age import AgeManager
from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import FSTSPSolution
from fstsp.formulation.stage_based import solution_components, solve_stage_model


def solve_cmsa(
    instance: FSTSPInstance,
    total_time: float = 120.0,
    mip_time: float = 15.0,
    age_limit: int = 2,
    truck_sample_ratio: float = 0.65,
    seed: int = 42,
    mip_rel_gap: float = 0.02,
    construct_tsp_time: float = 5.0,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> FSTSPSolution:
    """CMSA-style solver following Algorithm 1 at the architectural level.

    Each iteration constructs a certified feasible solution, merges its components,
    solves a restricted stage-based MIP, then adapts component ages. Construction
    details are a documented reimplementation choice because the authors' source
    code is not public.

    The implementation keeps the best certified constructed solution as a safety
    incumbent. This is important with SciPy/HiGHS because that backend does not
    expose a MIP-start API here; a very short restricted solve could otherwise fail
    to rediscover a known feasible construction.
    """
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    if mip_time <= 0:
        raise ValueError("mip_time must be positive")
    if construct_tsp_time < 0:
        raise ValueError("construct_tsp_time must be >= 0")
    if not 0 <= mip_rel_gap < 1:
        raise ValueError("mip_rel_gap must be in [0, 1)")

    rng = np.random.default_rng(seed)
    ages = AgeManager(age_limit)
    started = time.perf_counter()
    deadline = started + float(total_time)
    best: FSTSPSolution | None = None
    iteration = 0
    history: list[dict] = []

    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= total_time:
            break

        iteration += 1
        # Construction is part of the CMSA wall-clock budget.
        remaining_for_construct = max(0.0, deadline - time.perf_counter())
        tsp_budget = min(float(construct_tsp_time), remaining_for_construct)
        constructed = construct_solution(
            instance,
            rng,
            truck_sample_ratio,
            tsp_time_limit=tsp_budget,
            deadline=deadline,
        )
        if constructed.feasible:
            if best is None or (
                constructed.objective is not None
                and (best.objective is None or constructed.objective < best.objective)
            ):
                best = constructed

        c_comp = solution_components(constructed)
        ages.mark_useful(c_comp)
        active = ages.active()

        elapsed = time.perf_counter() - started
        remaining = total_time - elapsed
        if remaining <= 0:
            history.append(
                {
                    "iteration": iteration,
                    "constructed_objective": constructed.objective,
                    "restricted_feasible": False,
                    "restricted_objective": None,
                    "active_components": len(active),
                    "elapsed": elapsed,
                    "note": "time budget exhausted before restricted MIP",
                }
            )
            break

        local_limit = min(mip_time, remaining)
        mip_sol = solve_stage_model(
            instance,
            time_limit=local_limit,
            mip_rel_gap=mip_rel_gap,
            active_components=active,
            strengthen=True,
            deadline=deadline,
            solver_backend=solver_backend,
            threads=threads,
            mip_emphasis=mip_emphasis,
        )
        mip_comp = solution_components(mip_sol)
        if mip_sol.feasible:
            ages.mark_useful(mip_comp)
            if best is None or (
                mip_sol.objective is not None
                and (best.objective is None or mip_sol.objective < best.objective)
            ):
                best = mip_sol

        # Algorithm 1 Lines 11-18: adapt component ages across all active components.
        ages.adapt()
        history.append(
            {
                "iteration": iteration,
                "constructed_objective": constructed.objective,
                "construction_method": constructed.metadata.get("construction_method", "unknown"),
                "restricted_feasible": mip_sol.feasible,
                "restricted_objective": mip_sol.objective,
                "active_components": len(active),
                "n_variables": mip_sol.metadata.get("n_variables"),
                "n_constraints": mip_sol.metadata.get("n_constraints"),
                "n_nonzeros": mip_sol.metadata.get("n_nonzeros"),
                "n_fixed_zero_variables": mip_sol.metadata.get("n_fixed_zero_variables"),
                "n_fixed_one_variables": mip_sol.metadata.get("n_fixed_one_variables"),
                "n_free_variables": mip_sol.metadata.get("n_free_variables"),
                "n_active_constraints": mip_sol.metadata.get("n_active_constraints"),
                "n_active_nonzeros": mip_sol.metadata.get("n_active_nonzeros"),
                "presolved_variables": mip_sol.metadata.get("presolved_variables"),
                "presolved_constraints": mip_sol.metadata.get("presolved_constraints"),
                "presolved_nonzeros": mip_sol.metadata.get("presolved_nonzeros"),
                "solver_reported_variables": mip_sol.metadata.get("solver_reported_variables"),
                "solver_reported_constraints": mip_sol.metadata.get("solver_reported_constraints"),
                "solver_reported_nonzeros": mip_sol.metadata.get("solver_reported_nonzeros"),
                "elapsed": time.perf_counter() - started,
            }
        )

    if best is None:
        # The all-truck construction should make this path practically unreachable,
        # but keep a clear non-fabricated failure result if certification ever breaks.
        return FSTSPSolution(
            feasible=False,
            objective=None,
            runtime=time.perf_counter() - started,
            status="cmsa_no_certified_feasible_solution",
            metadata={"iterations": iteration, "history": history},
        )

    best.runtime = time.perf_counter() - started
    best.status = "cmsa_completed"
    best.metadata.update(
        {
            "iterations": iteration,
            "age_limit": age_limit,
            "mip_time": mip_time,
            "total_time_budget": total_time,
            "seed": seed,
            "solver_backend": solver_backend,
            "threads": threads,
            "mip_emphasis": mip_emphasis,
            "construct_tsp_time": construct_tsp_time,
            "budget_overrun_seconds": max(0.0, (time.perf_counter() - started) - total_time),
            "history": history,
            "last_mip_metadata": mip_sol.metadata if "mip_sol" in locals() else {},
        }
    )
    return best
