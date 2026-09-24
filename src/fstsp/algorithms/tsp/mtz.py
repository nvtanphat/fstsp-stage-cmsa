from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


@dataclass
class TSPResult:
    """Detailed result of an MTZ TSP solve."""

    route: list[int] | None
    status: str  # "OPTIMAL", "FEASIBLE", "TIME_LIMIT", "INFEASIBLE", "UNBOUNDED", "ERROR"
    proven_optimal: bool
    cost: float | None
    runtime: float
    raw_status: int | str
    message: str
    solver_backend: str = "highs"


def solve_tsp_mtz_result(
    nodes: list[int],
    distance: np.ndarray,
    start: int,
    end: int,
    time_limit: float = 10.0,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> TSPResult:
    """Solve directed Hamiltonian path start -> customers -> end with MTZ constraints, returning TSPResult."""
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if start == end:
        raise ValueError("start and end must be distinct node ids")
    customers = list(nodes)
    if len(customers) != len(set(customers)):
        raise ValueError("nodes must not contain duplicates")
    if start in customers or end in customers:
        raise ValueError("customer nodes must exclude start/end")

    all_nodes = [start] + customers + [end]
    arcs = [(i, j) for i in all_nodes for j in all_nodes if i != j and i != end and j != start]
    x_index = {a: idx for idx, a in enumerate(arcs)}
    offset = len(arcs)
    u_index = {h: offset + q for q, h in enumerate(customers)}
    nvar = offset + len(customers)

    c = np.zeros(nvar)
    for a, idx in x_index.items():
        c[idx] = distance[a]
    integrality = np.zeros(nvar, dtype=int)
    integrality[:offset] = 1
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    ub[:offset] = 1.0
    for h in customers:
        lb[u_index[h]] = 1.0
        ub[u_index[h]] = max(1, len(customers))

    rows, lows, highs = [], [], []

    def add(coeff, lo=-np.inf, hi=np.inf):
        rows.append(coeff)
        lows.append(lo)
        highs.append(hi)

    # start one outgoing; end one incoming
    add([(x_index[(i, j)], 1) for i, j in arcs if i == start], 1, 1)
    add([(x_index[(i, j)], 1) for i, j in arcs if j == end], 1, 1)
    for h in customers:
        add([(x_index[(i, j)], 1) for i, j in arcs if j == h], 1, 1)
        add([(x_index[(i, j)], 1) for i, j in arcs if i == h], 1, 1)
    n = max(1, len(customers))
    for i in customers:
        for j in customers:
            if i == j or (i, j) not in x_index:
                continue
            # u_i - u_j + n*x_ij <= n-1
            add([(u_index[i], 1), (u_index[j], -1), (x_index[(i, j)], n)], -np.inf, n - 1)

    A = lil_matrix((len(rows), nvar), dtype=float)
    for r, coeff in enumerate(rows):
        for idx, value in coeff:
            A[r, idx] += value

    from fstsp.formulation.stage_based import ModelData
    from fstsp.solver import get_solver_backend, SolverOptions

    model = ModelData(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraint=LinearConstraint(A.tocsr(), lows, highs),
        var=None,
        big_m=float(n),
    )
    backend = get_solver_backend(solver_backend)
    options = SolverOptions(
        time_limit=float(time_limit),
        threads=threads,
        mip_emphasis=mip_emphasis,
        presolve=True,
    )
    res = backend.solve(model, options)
    runtime = res.runtime
    raw_status = res.raw_status
    msg = res.message

    if res.x is None:
        if raw_status in (1, 107, 108):
            status = "TIME_LIMIT"
        elif raw_status in (2, 103):
            status = "INFEASIBLE"
        elif raw_status == 3:
            status = "UNBOUNDED"
        else:
            status = "ERROR"
        return TSPResult(
            route=None,
            status=status,
            proven_optimal=False,
            cost=None,
            runtime=runtime,
            raw_status=raw_status,
            message=msg,
            solver_backend=res.solver_name,
        )

    # Extract route from solution vector
    succ = {}
    for (i, j), idx in x_index.items():
        if res.x[idx] > 0.5:
            succ[i] = j
    route = [start]
    seen = {start}
    route_valid = True
    while route[-1] != end:
        nxt = succ.get(route[-1])
        if nxt is None or nxt in seen:
            route_valid = False
            break
        route.append(nxt)
        seen.add(nxt)
        if len(route) > len(all_nodes) + 1:
            route_valid = False
            break

    # Verify all customers are visited
    if len(route) != len(all_nodes) or route[-1] != end or not route_valid:
        return TSPResult(
            route=None,
            status="ERROR",
            proven_optimal=False,
            cost=None,
            runtime=runtime,
            raw_status=raw_status,
            message="Extracted tour is disconnected, subtoured, or incomplete",
            solver_backend=res.solver_name,
        )

    cost = float(res.objective) if res.objective is not None else sum(distance[u, v] for u, v in zip(route, route[1:]))

    if res.solver_name == "cplex":
        if raw_status in (101, 102) or (res.mip_gap is not None and res.mip_gap <= 1e-9):
            status = "OPTIMAL"
            proven_optimal = True
        else:
            status = "FEASIBLE"
            proven_optimal = False
    else:
        if raw_status == 0:
            status = "OPTIMAL"
            proven_optimal = True
        elif raw_status == 1:
            status = "FEASIBLE"
            proven_optimal = False
        else:
            status = "FEASIBLE"
            proven_optimal = False

    return TSPResult(
        route=route,
        status=status,
        proven_optimal=proven_optimal,
        cost=cost,
        runtime=runtime,
        raw_status=raw_status,
        message=msg,
        solver_backend=res.solver_name,
    )


def solve_tsp_mtz(
    nodes: list[int],
    distance: np.ndarray,
    start: int,
    end: int,
    time_limit: float = 10.0,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> list[int] | None:
    """Solve directed Hamiltonian path start -> customers -> end with MTZ constraints.

    Returns the tour as a node-id list if a feasible tour was found, or None otherwise.
    Backward-compatible convenience wrapper around solve_tsp_mtz_result.
    """
    result = solve_tsp_mtz_result(
        nodes,
        distance,
        start,
        end,
        time_limit=time_limit,
        solver_backend=solver_backend,
        threads=threads,
        mip_emphasis=mip_emphasis,
    )
    return result.route
