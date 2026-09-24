from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def solve_tsp_mtz(nodes: list[int], distance: np.ndarray, start: int, end: int, time_limit: float = 10.0) -> list[int] | None:
    """Solve a directed Hamiltonian path start -> customers -> end with MTZ constraints.

    `nodes` contains only customer node ids. This is used for small CMSA construction subsets.
    Returns None if the MILP does not produce a feasible solution within the limit.
    """
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
        rows.append(coeff); lows.append(lo); highs.append(hi)

    # start one outgoing; end one incoming
    add([(x_index[(i,j)],1) for i,j in arcs if i == start], 1, 1)
    add([(x_index[(i,j)],1) for i,j in arcs if j == end], 1, 1)
    for h in customers:
        add([(x_index[(i,j)],1) for i,j in arcs if j == h], 1, 1)
        add([(x_index[(i,j)],1) for i,j in arcs if i == h], 1, 1)
    n = max(1, len(customers))
    for i in customers:
        for j in customers:
            if i == j or (i,j) not in x_index:
                continue
            # u_i - u_j + n*x_ij <= n-1
            add([(u_index[i],1),(u_index[j],-1),(x_index[(i,j)],n)], -np.inf, n-1)

    A = lil_matrix((len(rows), nvar), dtype=float)
    for r, coeff in enumerate(rows):
        for idx, value in coeff:
            A[r, idx] += value
    res = milp(c, integrality=integrality, bounds=Bounds(lb, ub), constraints=LinearConstraint(A.tocsr(), lows, highs), options={"time_limit": time_limit, "presolve": True})
    if res.x is None:
        return None
    succ = {}
    for (i,j), idx in x_index.items():
        if res.x[idx] > 0.5:
            succ[i] = j
    route = [start]
    seen = {start}
    while route[-1] != end:
        nxt = succ.get(route[-1])
        if nxt is None or nxt in seen:
            return None
        route.append(nxt)
        seen.add(nxt)
        if len(route) > len(all_nodes) + 1:
            return None
    return route
