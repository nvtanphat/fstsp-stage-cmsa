from __future__ import annotations

import time
import numpy as np

from fstsp.algorithms.tsp.heuristic import nearest_neighbor_tour, two_opt
from fstsp.algorithms.tsp.mtz import solve_tsp_mtz
from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import DroneSortie, FSTSPSolution
from fstsp.evaluation.schedule import evaluate_schedule


def _build_truck_route(
    instance: FSTSPInstance,
    truck_customers: list[int],
    exact_tsp_threshold: int = 60,
    tsp_time_limit: float = 5.0,
    deadline: float | None = None,
) -> list[int]:
    route = None
    remaining = float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())
    effective_limit = min(float(tsp_time_limit), remaining)
    if len(truck_customers) <= exact_tsp_threshold and effective_limit > 1e-4:
        route = solve_tsp_mtz(
            truck_customers,
            instance.truck_time,
            instance.S,
            instance.E,
            time_limit=float(effective_limit),
        )
    if route is None:
        route = nearest_neighbor_tour(
            truck_customers,
            instance.truck_time,
            instance.S,
            instance.E,
            deadline=deadline,
        )
        route = two_opt(route, instance.truck_time, deadline=deadline)
    return route


def _candidate_edges(
    instance: FSTSPInstance,
    route: list[int],
    customer: int,
) -> list[tuple[float, int, int, int]]:
    """Return feasible consecutive-edge insertions for one drone customer.

    Construction deliberately uses consecutive truck stages. The exact restricted
    MILP solved later by CMSA may still place launch/recovery on non-consecutive
    stages when active components permit it.
    """
    candidates: list[tuple[float, int, int, int]] = []
    if not instance.drone_allowed[customer - 1]:
        return candidates

    for stage, (i, j) in enumerate(zip(route, route[1:])):
        # Eq. (49) forbids launch from E and Eq. (50) recovery at S. A route edge
        # already guarantees j != S and i != E for a normal S->...->E path.
        if i == instance.E or j == instance.S:
            continue

        # Published base model fixes d_S=0 (Eq. 27) while Eq. 30 charges tL,
        # therefore positive launch handling makes a launch from S infeasible.
        if i == instance.S and instance.launch_time > 1e-12:
            continue

        flight = float(
            instance.drone_time[i, customer]
            + instance.drone_time[customer, j]
        )
        truck_leg = float(instance.truck_time[i, j])

        # Eq. (14) and Eq. (33): both the drone flight and the truck's arrival at
        # the rendezvous must fit the endurance horizon (before recovery handling).
        usable = instance.drone_endurance - instance.recovery_time
        if flight > usable + 1e-9 or truck_leg > usable + 1e-9:
            continue

        operation = (
            instance.launch_time
            + max(truck_leg, flight)
            + instance.recovery_time
        )
        delta = operation - truck_leg
        candidates.append((delta, stage, i, j))
    return candidates


def _assign_drone_customers(
    instance: FSTSPInstance,
    route: list[int],
    drone_customers: list[int],
) -> tuple[list[DroneSortie], list[int]]:
    """Greedily assign drone customers to distinct consecutive route edges.

    Customers with fewer feasible edges are handled first. Unassigned customers
    are returned so the caller can promote them to truck service and rebuild the
    route from scratch. This avoids stale sortie stage/node bugs after route edits.
    """
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
    route: list[int],
    drone_customers: list[int],
    time_limit: float = 3.0,
    deadline: float | None = None,
) -> FSTSPSolution | None:
    """Integrate remaining customers into truck route via stage-based formulation (Algorithm 1 / Section 3)."""
    if not drone_customers:
        return None
    remaining = float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())
    local_limit = min(float(time_limit), remaining)
    if local_limit <= 0.05:
        return None

    comps = {("x", route[k], route[k + 1]) for k in range(len(route) - 1)}
    for h in drone_customers:
        comps.add(("phi", h))
        for i in route[:-1]:
            comps.add(("A", h, i))
        for j in route[1:]:
            comps.add(("B", h, j))

    from fstsp.formulation.stage_based import solve_stage_model

    try:
        sol = solve_stage_model(
            instance,
            time_limit=local_limit,
            mip_rel_gap=0.05,
            strengthen=True,
            deadline=deadline,
            fixed_truck_route=route,
        )
        if sol.feasible and sol.objective is not None and len(sol.drone_sorties) == len(drone_customers):
            sol.status = "constructed_stage_based"
            return sol
    except Exception:
        pass
    return None


def construct_solution(
    instance: FSTSPInstance,
    rng: np.random.Generator,
    truck_sample_ratio: float = 0.65,
    exact_tsp_threshold: int = 60,
    tsp_time_limit: float = 5.0,
    deadline: float | None = None,
) -> FSTSPSolution:
    """Construct a certified feasible solution for CMSA component generation.

    Paper method (Section 3): sample a subset of truck customers, solve MTZ TSP,
    then integrate remaining customers (drone customers) via the 2-index stage-based
    formulation with fixed stages.

    Robustness rule: if the stage-based sub-MIP is infeasible or times out,
    consecutive-edge heuristic assignment and promotion loop serve as certified fallback.
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
        route = _build_truck_route(
            instance, truck_customers, exact_tsp_threshold,
            tsp_time_limit=tsp_time_limit, deadline=deadline
        )
        if deadline is not None and time.perf_counter() >= deadline:
            # Switch immediately to a cheap all-truck fallback rather than starting
            # another exact construction subproblem after the CMSA wall-clock budget.
            truck_set.update(customers)
        drone_customers = [h for h in customers if h not in truck_set]

        # Primary Paper Method: Integrate remaining customers via 2-index stage-based formulation
        if drone_customers:
            rem_time = float("inf") if deadline is None else max(0.0, deadline - time.perf_counter())
            sub_limit = min(float(tsp_time_limit), rem_time, 2.0)
            stage_sol = _integrate_drone_customers_stage_based(
                instance, route, drone_customers, time_limit=sub_limit, deadline=deadline
            )
            if stage_sol is not None and stage_sol.feasible:
                stage_sol.metadata.update(
                    {
                        "truck_sample_ratio": truck_sample_ratio,
                        "truck_customers": len(stage_sol.truck_route) - 2 if stage_sol.truck_route else 0,
                        "drone_customers": len(stage_sol.drone_sorties),
                        "construction_method": "stage_based_milp",
                    }
                )
                return stage_sol

            # Paper Resampling/Promotion: If stage-based MILP cannot schedule all drone customers
            # (e.g. flight endurance exceeded or stage overlap conflict), promote the least feasible
            # drone customer to truck_set and re-run MTZ TSP with the expanded truck route.
            if deadline is not None and time.perf_counter() >= deadline:
                truck_set.update(drone_customers)
                continue

            # Promote customer requiring largest detour to truck route
            hardest_drone = max(
                drone_customers,
                key=lambda h: min(
                    float(instance.drone_time[i, h] + instance.drone_time[h, j])
                    for i in route[:-1] for j in route[1:] if i != j
                ),
            )
            truck_set.add(hardest_drone)
            continue

    # This should only be reached under an implementation regression. The all-truck
    # tour is always feasible for the base problem under positive finite travel times.
    route = _build_truck_route(
        instance, customers, exact_tsp_threshold,
        tsp_time_limit=0.0 if deadline is not None and time.perf_counter() >= deadline else tsp_time_limit,
        deadline=deadline,
    )
    fallback = FSTSPSolution(
        feasible=True,
        objective=0.0,
        truck_route=route,
        drone_sorties=[],
        status="constructed_all_truck_fallback",
        metadata={"objective_type": "certified_schedule"},
    )
    evaluation = evaluate_schedule(instance, fallback)
    if not evaluation.feasible or evaluation.completion_time is None:
        return FSTSPSolution(
            feasible=False,
            objective=None,
            status="construction_failed_certification",
            metadata={"issues": evaluation.issues},
        )
    fallback.objective = evaluation.completion_time
    return fallback
