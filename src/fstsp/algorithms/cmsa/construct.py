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
    exact_tsp_threshold: int,
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
        )
        route = two_opt(route, instance.truck_time)
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


def construct_solution(
    instance: FSTSPInstance,
    rng: np.random.Generator,
    truck_sample_ratio: float = 0.65,
    exact_tsp_threshold: int = 12,
    tsp_time_limit: float = 5.0,
    deadline: float | None = None,
) -> FSTSPSolution:
    """Construct a certified feasible solution for CMSA component generation.

    Paper-defined idea: sample truck customers, solve a TSP, then integrate the
    remaining customers as drone customers. The exact authors' construction code
    is not public, so the assignment policy is an explicit reimplementation choice.

    Robustness rule: whenever a drone customer cannot be placed, it is promoted to
    truck service and *all* sorties are recomputed against the rebuilt route. This
    prevents stale launch/recovery stages after route changes.
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
        sorties, failed = _assign_drone_customers(instance, route, drone_customers)
        if failed:
            truck_set.update(failed)
            continue

        provisional = FSTSPSolution(
            feasible=True,
            objective=0.0,
            truck_route=route,
            drone_sorties=sorties,
            status="constructed",
            metadata={"objective_type": "certified_schedule"},
        )
        evaluation = evaluate_schedule(instance, provisional)
        if not evaluation.feasible or evaluation.completion_time is None:
            # Defensive fallback: if our construction policy ever produces a plan
            # outside the published model, promote all drone customers to truck.
            truck_set.update(s.customer for s in sorties)
            continue

        provisional.objective = evaluation.completion_time
        provisional.metadata.update(
            {
                "truck_sample_ratio": truck_sample_ratio,
                "truck_customers": len(route) - 2,
                "drone_customers": len(sorties),
            }
        )
        return provisional

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
