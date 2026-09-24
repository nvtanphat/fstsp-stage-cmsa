from __future__ import annotations

from dataclasses import dataclass, field
import math

from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import FSTSPSolution


@dataclass
class ScheduleEvaluation:
    """Deterministic feasibility/timing reconstruction for an FSTSP solution.

    The reconstruction mirrors the paper's timing logic for the base formulation:
    recovery is processed before a possible launch at the same truck stage, the
    truck/drone synchronize at recovery, and drone endurance covers both flight
    time and waiting for the truck at the recovery node.
    """

    feasible: bool
    completion_time: float | None
    issues: list[str] = field(default_factory=list)
    truck_arrival: list[float] = field(default_factory=list)
    truck_departure: list[float] = field(default_factory=list)


def evaluate_schedule(instance: FSTSPInstance, solution: FSTSPSolution) -> ScheduleEvaluation:
    issues: list[str] = []
    route = list(solution.truck_route)
    if not route:
        return ScheduleEvaluation(False, None, ["empty truck route"])
    if route[0] != instance.S:
        issues.append("truck route does not start at S")
    if route[-1] != instance.E:
        issues.append("truck route does not end at E")
    if len(route) > instance.n + 2:
        issues.append("truck route exceeds the N+2 stage horizon")

    valid_nodes = set(instance.nodes)
    if any(node not in valid_nodes for node in route):
        issues.append("truck route contains an invalid node id")
    if instance.S in route[1:]:
        issues.append("start depot S appears after stage 1")
    if instance.E in route[:-1]:
        issues.append("end depot E appears before the final truck stage")

    # The base model does not support revisits. S and E are separate node ids even
    # though they share coordinates, so any duplicate id is a true revisit.
    if len(route) != len(set(route)):
        issues.append("truck route revisits a node; base formulation has no revisit")

    positions = {node: stage for stage, node in enumerate(route)}
    launch_by_stage: dict[int, object] = {}
    recovery_by_stage: dict[int, object] = {}
    intervals: list[tuple[int, int, int]] = []

    for sortie in solution.drone_sorties:
        h = sortie.customer
        if h not in instance.customers:
            issues.append(f"sortie has invalid customer {h}")
            continue
        if not instance.drone_allowed[h - 1]:
            issues.append(f"drone serves forbidden customer {h}")
        if sortie.launch_node == instance.E:
            issues.append(f"sortie {h} launches from E, forbidden by Eq. (49)")
        if sortie.recovery_node == instance.S:
            issues.append(f"sortie {h} recovers at S, forbidden by Eq. (50)")
        if sortie.launch_node not in positions or sortie.recovery_node not in positions:
            issues.append(f"sortie {h} launch/recovery not on truck route")
            continue

        ls = positions[sortie.launch_node]
        rs = positions[sortie.recovery_node]
        if sortie.launch_stage is not None and sortie.launch_stage != ls:
            issues.append(
                f"sortie {h} launch_stage={sortie.launch_stage} does not match route stage {ls}"
            )
        if sortie.recovery_stage is not None and sortie.recovery_stage != rs:
            issues.append(
                f"sortie {h} recovery_stage={sortie.recovery_stage} does not match route stage {rs}"
            )
        if ls >= rs:
            issues.append(f"sortie {h} recovery does not follow launch")
            continue
        if ls in launch_by_stage:
            issues.append(f"more than one drone launch at truck stage {ls}")
        else:
            launch_by_stage[ls] = sortie
        if rs in recovery_by_stage:
            issues.append(f"more than one drone recovery at truck stage {rs}")
        else:
            recovery_by_stage[rs] = sortie
        intervals.append((ls, rs, h))

    # Eq. (15): at any cut between consecutive truck stages, at most one sortie
    # can be in flight. Adjacent sorties [a,b) and [b,c) are allowed.
    for cut in range(max(0, len(route) - 1)):
        active = [h for ls, rs, h in intervals if ls <= cut < rs]
        if len(active) > 1:
            issues.append(f"overlapping drone sorties across stage cut {cut}: {active}")

    truck_customers = [i for i in route if i in instance.customers]
    drone_customers = [s.customer for s in solution.drone_sorties]
    served = truck_customers + drone_customers
    if sorted(served) != sorted(instance.customers):
        issues.append(f"customer coverage mismatch: served={sorted(served)}")
    if len(served) != len(set(served)):
        issues.append("a customer is served more than once")

    # The paper fixes a_S = d_S = 0 (Eq. 27) while Eq. (30) charges launch time.
    # Therefore a positive launch time makes a launch from S infeasible in the
    # published base formulation.
    if 0 in launch_by_stage and instance.launch_time > 1e-12:
        h = launch_by_stage[0].customer
        issues.append(
            f"sortie {h} launches from S with positive launch_time, conflicting with Eqs. (27),(30)"
        )

    if issues:
        return ScheduleEvaluation(False, None, issues)

    arrivals = [0.0] * len(route)
    departures = [0.0] * len(route)
    launch_departure: dict[int, float] = {}
    drone_arrival: dict[int, float] = {}

    for stage, node in enumerate(route):
        if stage == 0:
            arrivals[stage] = 0.0
        else:
            prev = route[stage - 1]
            arrivals[stage] = departures[stage - 1] + float(instance.truck_time[prev, node])

        depart = arrivals[stage]

        recovery = recovery_by_stage.get(stage)
        if recovery is not None:
            h = recovery.customer
            launch_t = launch_departure[h]
            flight = float(
                instance.drone_time[recovery.launch_node, h]
                + instance.drone_time[h, recovery.recovery_node]
            )
            drone_arrival[h] = launch_t + flight

            # Eq. (14): flight plus recovery handling must fit endurance.
            if flight + instance.recovery_time > instance.drone_endurance + 1e-7:
                issues.append(f"sortie {h} violates drone flight endurance (Eq. 14)")

            # Eq. (33): the truck must reach the rendezvous before endurance expires;
            # otherwise the drone would have to remain airborne too long.
            if (
                arrivals[stage] - launch_t
                > instance.drone_endurance - instance.recovery_time + 1e-7
            ):
                issues.append(f"sortie {h} violates rendezvous endurance (Eq. 33)")

            depart = max(depart, drone_arrival[h]) + instance.recovery_time

        launch = launch_by_stage.get(stage)
        if launch is not None:
            # If recovery and launch share a stage, recovery occurs first, matching
            # Eqs. (30)-(32), then launch handling is charged.
            depart += instance.launch_time
            launch_departure[launch.customer] = depart

        departures[stage] = depart

    completion = departures[-1]
    if not math.isfinite(completion):
        issues.append("non-finite reconstructed completion time")

    return ScheduleEvaluation(
        feasible=not issues,
        completion_time=None if issues else float(completion),
        issues=issues,
        truck_arrival=arrivals,
        truck_departure=departures,
    )
