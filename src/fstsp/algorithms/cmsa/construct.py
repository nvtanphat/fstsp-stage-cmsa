from __future__ import annotations

import logging
import time
import numpy as np

from fstsp.algorithms.tsp.heuristic import nearest_neighbor_tour, two_opt
from fstsp.algorithms.tsp.mtz import solve_tsp_mtz_result
from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import DroneSortie, FSTSPSolution
from fstsp.evaluation.schedule import evaluate_schedule

logger = logging.getLogger(__name__)


def _build_truck_route(
    instance: FSTSPInstance,
    truck_customers: list[int],
    exact_tsp_threshold: int = 60,
    tsp_time_limit: float = 5.0,
    deadline: float | None = None,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> tuple[list[int], str, bool, str]:
    """Solve or approximate TSP for truck customers, returning (route, status, proven_optimal, actual_solver)."""
    route = None
    status = "UNKNOWN"
    proven_optimal = False
    actual_solver = "heuristic"
    remaining = float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())
    effective_limit = min(float(tsp_time_limit), remaining)

    if len(truck_customers) <= exact_tsp_threshold and effective_limit > 1e-4:
        tsp_res = solve_tsp_mtz_result(
            truck_customers,
            instance.truck_time,
            instance.S,
            instance.E,
            time_limit=float(effective_limit),
            solver_backend=solver_backend,
            threads=threads,
            mip_emphasis=mip_emphasis,
        )
        if tsp_res.route is not None:
            route = tsp_res.route
            status = tsp_res.status
            proven_optimal = tsp_res.proven_optimal
            actual_solver = tsp_res.solver_backend

    if route is None:
        route = nearest_neighbor_tour(
            truck_customers,
            instance.truck_time,
            instance.S,
            instance.E,
            deadline=deadline,
        )
        route = two_opt(route, instance.truck_time, deadline=deadline)
        status = "HEURISTIC_2OPT"
        proven_optimal = False
        actual_solver = "heuristic_2opt"

    return route, status, proven_optimal, actual_solver


def _candidate_edges(
    instance: FSTSPInstance,
    route: list[int],
    customer: int,
) -> list[tuple[float, int, int, int]]:
    """Return feasible consecutive-edge insertions for one drone customer."""
    candidates: list[tuple[float, int, int, int]] = []
    if not instance.drone_allowed[customer - 1]:
        return candidates

    for stage, (i, j) in enumerate(zip(route, route[1:])):
        if i == instance.E or j == instance.S:
            continue
        if i == instance.S and instance.launch_time > 1e-12:
            continue

        flight = float(instance.drone_time[i, customer] + instance.drone_time[customer, j])
        truck_leg = float(instance.truck_time[i, j])
        usable = instance.drone_endurance - instance.recovery_time
        if flight > usable + 1e-9 or truck_leg > usable + 1e-9:
            continue

        operation = instance.launch_time + max(truck_leg, flight) + instance.recovery_time
        delta = operation - truck_leg
        candidates.append((delta, stage, i, j))
    return candidates


def _assign_drone_customers(
    instance: FSTSPInstance,
    route: list[int],
    drone_customers: list[int],
) -> tuple[list[DroneSortie], list[int]]:
    """Greedily assign drone customers to distinct consecutive route edges (heuristic fallback)."""
    options = {
        h: sorted(_candidate_edges(instance, route, h), key=lambda x: (x[0], x[1]))
        for h in drone_customers
    }
    ordered = sorted(drone_customers, key=lambda h: (len(options[h]), options[h][0][0] if options[h] else np.inf, h))

    used_edges: set[int] = set()
    sorties: list[DroneSortie] = []
    failed: list[int] = []
    for h in ordered:
        chosen = next((c for c in options[h] if c[1] not in used_edges), None)
        if chosen is None:
            failed.append(h)
            continue
        _, stage, i, j = chosen
        sorties.append(
            DroneSortie(
                launch_node=i,
                customer=h,
                recovery_node=j,
                launch_stage=stage,
                recovery_stage=stage + 1,
            )
        )
        used_edges.add(stage)

    sorties.sort(key=lambda s: (s.launch_stage if s.launch_stage is not None else -1, s.customer))
    return sorties, failed


def _integrate_drone_customers_stage_based(
    instance: FSTSPInstance,
    truck_customers: list[int],
    drone_customers: list[int],
    route: list[int] | None = None,
    time_limit: float = 3.0,
    deadline: float | None = None,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> FSTSPSolution | None:
    """Integrate remaining customers via 2-index stage-based MILP with fixed number of stages.

    Paper Section 3: Uses fixed number of stages equal to TSP tour length plus 2 (K = |C_truck| + 2).
    First attempts flexible truck ordering among truck customers so truck can adjust route for drone.
    If flexible solve times out, falls back to locked truck route sequence from MTZ TSP.
    """
    if not drone_customers:
        return None
    remaining = float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())
    local_limit = min(float(time_limit), remaining)
    if local_limit <= 0.05:
        return None

    from fstsp.formulation.stage_based import solve_stage_model

    k_stages = len(truck_customers) + 2

    # Attempt 1: Flexible truck ordering with fixed stages K and fixed customer sets
    # This allows the truck to reorder truck customers to optimally coordinate with drone sorties.
    try:
        sub_limit_flex = min(local_limit * 0.6, 2.0)
        sol = solve_stage_model(
            instance,
            time_limit=sub_limit_flex,
            mip_rel_gap=0.05,
            strengthen=True,
            deadline=deadline,
            num_stages=k_stages,
            fixed_truck_customers=truck_customers,
            fixed_drone_customers=drone_customers,
            solver_backend=solver_backend,
            threads=threads,
            mip_emphasis=mip_emphasis,
        )
        if sol.feasible and sol.objective is not None and len(sol.drone_sorties) == len(drone_customers):
            sol.status = "constructed_stage_based"
            sol.metadata["construction_method"] = "stage_based_milp"
            sol.metadata["stage_integration_mode"] = "flexible_truck_route"
            sol.metadata["construction_integration_backend"] = solver_backend
            return sol
    except Exception as exc:
        logger.debug("Flexible stage-based integration exception: %s", exc)

    # Attempt 2: Fixed truck route sequence from MTZ TSP
    # Fast restricted solve with locked truck route arcs if flexible solve timed out
    if route is not None:
        rem = float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())
        sub_limit_fix = min(local_limit * 0.4, rem, 1.5)
        if sub_limit_fix > 0.05:
            try:
                sol = solve_stage_model(
                    instance,
                    time_limit=sub_limit_fix,
                    mip_rel_gap=0.05,
                    strengthen=True,
                    deadline=deadline,
                    fixed_truck_route=route,
                    solver_backend=solver_backend,
                    threads=threads,
                    mip_emphasis=mip_emphasis,
                )
                if sol.feasible and sol.objective is not None and len(sol.drone_sorties) == len(drone_customers):
                    sol.status = "constructed_stage_based"
                    sol.metadata["construction_method"] = "stage_based_milp"
                    sol.metadata["stage_integration_mode"] = "fixed_truck_route"
                    sol.metadata["construction_integration_backend"] = solver_backend
                    return sol
            except Exception as exc:
                logger.debug("Fixed-route stage-based integration exception: %s", exc)

    return None


def construct_solution(
    instance: FSTSPInstance,
    rng: np.random.Generator,
    truck_sample_ratio: float = 0.65,
    exact_tsp_threshold: int = 60,
    tsp_time_limit: float = 5.0,
    deadline: float | None = None,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> FSTSPSolution:
    """Construct a certified feasible solution for CMSA component generation.

    Paper method (Section 3): sample a subset of truck customers, solve MTZ TSP,
    then integrate remaining customers (drone customers) via the 2-index stage-based
    formulation with fixed number of stages equal to TSP tour length plus 2.

    Explicit Fallback Hierarchy:
    1. stage_based_milp: Solves stage-based MILP with K = |C_truck| + 2 stages.
    2. heuristic_fallback: Consecutive-edge greedy assignment if MILP fails.
    3. all_truck_fallback: All-truck tour if no drone assignment succeeds.

    The actual method used is always recorded in solution.status and
    solution.metadata["construction_method"].
    """
    if not 0.0 < truck_sample_ratio <= 1.0:
        raise ValueError("truck_sample_ratio must be in (0, 1]")
    if exact_tsp_threshold < 0:
        raise ValueError("exact_tsp_threshold must be >= 0")
    if tsp_time_limit < 0:
        raise ValueError("tsp_time_limit must be >= 0")

    customers = instance.customers
    target = min(len(customers), max(1, int(round(len(customers) * truck_sample_ratio))))
    truck_set = set(rng.choice(customers, size=target, replace=False).tolist())

    # NOVISIT/forbidden drone customers are mandatory truck customers.
    truck_set.update(h for h in customers if not instance.drone_allowed[h - 1])

    # At most N iterations: each unsuccessful pass promotes at least one customer.
    for _ in range(instance.n + 1):
        truck_customers = sorted(truck_set)
        route, mtz_status, mtz_optimal, actual_tsp_backend = _build_truck_route(
            instance,
            truck_customers,
            exact_tsp_threshold,
            tsp_time_limit=tsp_time_limit,
            deadline=deadline,
            solver_backend=solver_backend,
            threads=threads,
            mip_emphasis=mip_emphasis,
        )
        if deadline is not None and time.perf_counter() >= deadline:
            truck_set.update(customers)
        drone_customers = [h for h in customers if h not in truck_set]

        # Primary Paper Method: Integrate remaining customers via 2-index stage-based formulation
        if drone_customers:
            rem_time = float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())
            sub_limit = min(float(tsp_time_limit), rem_time, 2.5)
            stage_sol = _integrate_drone_customers_stage_based(
                instance,
                truck_customers,
                drone_customers,
                route=route,
                time_limit=sub_limit,
                deadline=deadline,
                solver_backend=solver_backend,
                threads=threads,
                mip_emphasis=mip_emphasis,
            )
            if stage_sol is not None and stage_sol.feasible:
                stage_sol.metadata.update(
                    {
                        "truck_sample_ratio": truck_sample_ratio,
                        "truck_customers": len(stage_sol.truck_route) - 2 if stage_sol.truck_route else 0,
                        "drone_customers": len(stage_sol.drone_sorties),
                        "construction_method": "stage_based_milp",
                        "construction_tsp_backend": actual_tsp_backend,
                        "construction_integration_backend": solver_backend,
                        "solver_backend": solver_backend,
                        "mtz_status": mtz_status,
                        "mtz_proven_optimal": mtz_optimal,
                    }
                )
                return stage_sol

            # Heuristic assignment check before promotion
            heur_sorties, heur_failed = _assign_drone_customers(instance, route, drone_customers)
            if not heur_failed and heur_sorties:
                heur_cand = FSTSPSolution(
                    feasible=True,
                    objective=0.0,
                    truck_route=route,
                    drone_sorties=heur_sorties,
                    status="constructed_heuristic_fallback",
                    metadata={
                        "truck_sample_ratio": truck_sample_ratio,
                        "truck_customers": len(route) - 2,
                        "drone_customers": len(heur_sorties),
                        "construction_method": "heuristic_fallback",
                        "construction_tsp_backend": actual_tsp_backend,
                        "construction_integration_backend": "heuristic_greedy",
                        "solver_backend": solver_backend,
                        "mtz_status": mtz_status,
                        "mtz_proven_optimal": mtz_optimal,
                    },
                )
                cert = evaluate_schedule(instance, heur_cand)
                if cert.feasible and cert.completion_time is not None:
                    heur_cand.objective = cert.completion_time
                    return heur_cand

            # Paper Resampling/Promotion: If stage-based MILP cannot schedule all drone customers,
            # promote customer requiring largest detour to truck_set and re-run MTZ TSP.
            if deadline is not None and time.perf_counter() >= deadline:
                truck_set.update(drone_customers)
                continue

            hardest_drone = max(
                drone_customers,
                key=lambda h: min(
                    float(instance.drone_time[i, h] + instance.drone_time[h, j])
                    for i in route[:-1]
                    for j in route[1:]
                    if i != j
                ),
            )
            truck_set.add(hardest_drone)
            continue

    # All-truck fallback
    route, mtz_status, mtz_optimal, actual_tsp_backend = _build_truck_route(
        instance,
        customers,
        exact_tsp_threshold,
        tsp_time_limit=0.0 if deadline is not None and time.perf_counter() >= deadline else tsp_time_limit,
        deadline=deadline,
        solver_backend=solver_backend,
        threads=threads,
        mip_emphasis=mip_emphasis,
    )
    fallback = FSTSPSolution(
        feasible=True,
        objective=0.0,
        truck_route=route,
        drone_sorties=[],
        status="constructed_all_truck_fallback",
        metadata={
            "construction_method": "all_truck_fallback",
            "construction_tsp_backend": actual_tsp_backend,
            "construction_integration_backend": "none_all_truck",
            "solver_backend": solver_backend,
            "objective_type": "certified_schedule",
            "mtz_status": mtz_status,
            "mtz_proven_optimal": mtz_optimal,
        },
    )
    evaluation = evaluate_schedule(instance, fallback)
    if not evaluation.feasible or evaluation.completion_time is None:
        return FSTSPSolution(
            feasible=False,
            objective=None,
            status="construction_failed_certification",
            metadata={"issues": evaluation.issues, "construction_method": "failed"},
        )
    fallback.objective = evaluation.completion_time
    return fallback
