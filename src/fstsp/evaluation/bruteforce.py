from __future__ import annotations

import itertools
import math

from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import DroneSortie, FSTSPSolution


def _oracle_completion_time(
    instance: FSTSPInstance,
    route: list[int],
    sorties: list[DroneSortie],
) -> float | None:
    """Small-instance timing oracle intentionally separate from schedule.py.

    This duplicates the base-model semantics instead of calling the production
    validator, reducing common-mode errors in exact-vs-brute-force regression.
    It returns None for an infeasible discrete plan.
    """
    if not route or route[0] != instance.S or route[-1] != instance.E:
        return None
    if len(route) != len(set(route)) or len(route) > instance.n + 2:
        return None

    positions = {node: k for k, node in enumerate(route)}
    if set(positions) - set(instance.nodes):
        return None

    truck_customers = [x for x in route if x in instance.customers]
    drone_customers = [s.customer for s in sorties]
    served = truck_customers + drone_customers
    if sorted(served) != sorted(instance.customers) or len(served) != len(set(served)):
        return None

    launch_at: dict[int, DroneSortie] = {}
    recovery_at: dict[int, DroneSortie] = {}
    intervals: list[tuple[int, int]] = []
    for s in sorties:
        if s.customer not in instance.customers or not instance.drone_allowed[s.customer - 1]:
            return None
        if s.launch_node not in positions or s.recovery_node not in positions:
            return None
        ls, rs = positions[s.launch_node], positions[s.recovery_node]
        if s.launch_stage is not None and s.launch_stage != ls:
            return None
        if s.recovery_stage is not None and s.recovery_stage != rs:
            return None
        if ls >= rs or s.launch_node == instance.E or s.recovery_node == instance.S:
            return None
        if ls in launch_at or rs in recovery_at:
            return None
        if ls == 0 and instance.launch_time > 1e-12:
            return None
        if any(max(ls, a) < min(rs, b) for a, b in intervals):
            return None
        launch_at[ls] = s
        recovery_at[rs] = s
        intervals.append((ls, rs))

    arrival = [0.0] * len(route)
    departure = [0.0] * len(route)
    launch_departure: dict[int, float] = {}

    for k, node in enumerate(route):
        if k:
            arrival[k] = departure[k - 1] + float(instance.truck_time[route[k - 1], node])
        depart = arrival[k]

        r = recovery_at.get(k)
        if r is not None:
            if r.customer not in launch_departure:
                return None
            start_t = launch_departure[r.customer]
            flight = float(
                instance.drone_time[r.launch_node, r.customer]
                + instance.drone_time[r.customer, r.recovery_node]
            )
            if flight + instance.recovery_time > instance.drone_endurance + 1e-8:
                return None
            if arrival[k] - start_t > instance.drone_endurance - instance.recovery_time + 1e-8:
                return None
            drone_arrival = start_t + flight
            depart = max(depart, drone_arrival) + instance.recovery_time

        launch = launch_at.get(k)
        if launch is not None:
            depart += instance.launch_time
            launch_departure[launch.customer] = depart

        departure[k] = depart

    value = departure[-1]
    return float(value) if math.isfinite(value) else None


def brute_force_optimum(
    instance: FSTSPInstance,
    *,
    max_customers: int = 7,
) -> FSTSPSolution:
    """Independent tiny-instance oracle for regression testing.

    Enumerates truck-customer subsets/orders and non-overlapping drone sorties.
    It is intentionally exponential and must only be used for very small instances.
    The objective is evaluated by `_oracle_completion_time`, not by the production
    schedule validator used to certify MILP outputs.
    """
    if instance.n > max_customers:
        raise ValueError(
            f"brute_force_optimum is limited to n<={max_customers}; got n={instance.n}"
        )

    customers = instance.customers
    best_value = math.inf
    best_solution: FSTSPSolution | None = None

    for mask in range(1 << len(customers)):
        truck_customers = [
            customers[q] for q in range(len(customers)) if (mask >> q) & 1
        ]
        drone_customers = [h for h in customers if h not in truck_customers]
        if any(not instance.drone_allowed[h - 1] for h in drone_customers):
            continue

        for order in itertools.permutations(truck_customers):
            route = [instance.S, *order, instance.E]
            if not drone_customers:
                value = _oracle_completion_time(instance, route, [])
                if value is not None and value < best_value:
                    best_value = value
                    best_solution = FSTSPSolution(
                        feasible=True,
                        objective=value,
                        truck_route=route,
                        drone_sorties=[],
                        status="bruteforce",
                    )
                continue

            pairs = [(k, kp) for k in range(len(route)) for kp in range(k + 1, len(route))]
            assignments: list[DroneSortie] = []
            used_launch: set[int] = set()
            used_recovery: set[int] = set()
            intervals: list[tuple[int, int]] = []

            def search(idx: int) -> None:
                nonlocal best_value, best_solution
                if idx == len(drone_customers):
                    value = _oracle_completion_time(instance, route, assignments)
                    if value is not None and value < best_value - 1e-10:
                        best_value = value
                        best_solution = FSTSPSolution(
                            feasible=True,
                            objective=value,
                            truck_route=list(route),
                            drone_sorties=list(assignments),
                            status="bruteforce",
                        )
                    return

                h = drone_customers[idx]
                for k, kp in pairs:
                    if k in used_launch or kp in used_recovery:
                        continue
                    if any(max(k, a) < min(kp, b) for a, b in intervals):
                        continue
                    i, j = route[k], route[kp]
                    if i == instance.E or j == instance.S:
                        continue
                    if i == instance.S and instance.launch_time > 1e-12:
                        continue
                    flight = float(instance.drone_time[i, h] + instance.drone_time[h, j])
                    if flight + instance.recovery_time > instance.drone_endurance + 1e-9:
                        continue

                    assignments.append(DroneSortie(i, h, j, k, kp))
                    used_launch.add(k)
                    used_recovery.add(kp)
                    intervals.append((k, kp))
                    search(idx + 1)
                    intervals.pop()
                    used_recovery.remove(kp)
                    used_launch.remove(k)
                    assignments.pop()

            search(0)

    if best_solution is None:
        return FSTSPSolution(
            feasible=False,
            objective=None,
            status="bruteforce_no_feasible_solution",
        )
    return best_solution
