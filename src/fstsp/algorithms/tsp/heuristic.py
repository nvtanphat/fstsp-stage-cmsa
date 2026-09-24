from __future__ import annotations

import time
import numpy as np


def tour_cost(route: list[int], distance: np.ndarray) -> float:
    return float(sum(distance[a, b] for a, b in zip(route, route[1:])))


def nearest_neighbor_tour(
    nodes: list[int],
    distance: np.ndarray,
    start: int,
    end: int,
    deadline: float | None = None,
) -> list[int]:
    remaining = set(nodes)
    route = [start]
    current = start
    while remaining:
        if deadline is not None and time.perf_counter() >= deadline:
            route.extend(sorted(remaining))
            break
        nxt = min(remaining, key=lambda j: (distance[current, j], j))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    route.append(end)
    return route


def two_opt(
    route: list[int],
    distance: np.ndarray,
    max_passes: int = 20,
    deadline: float | None = None,
) -> list[int]:
    best = route[:]
    best_cost = tour_cost(best, distance)
    for _ in range(max_passes):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        improved = False
        for i in range(1, len(best) - 2):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                cost = tour_cost(candidate, distance)
                if cost + 1e-9 < best_cost:
                    best, best_cost = candidate, cost
                    improved = True
        if not improved:
            break
    return best
