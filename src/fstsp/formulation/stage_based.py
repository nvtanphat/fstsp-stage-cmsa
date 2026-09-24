from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import DroneSortie, FSTSPSolution


Component = tuple


class ModelBuildTimeout(RuntimeError):
    """Raised when a caller-supplied absolute deadline expires during model assembly."""


class VarIndex:
    def __init__(self) -> None:
        self._next = 0
        self._map: dict[tuple, int] = {}

    def add(self, key: tuple) -> int:
        idx = self._next
        self._map[key] = idx
        self._next += 1
        return idx

    def __getitem__(self, key: tuple) -> int:
        return self._map[key]

    def get(self, key: tuple) -> int | None:
        return self._map.get(key)

    @property
    def size(self) -> int:
        return self._next


@dataclass
class ModelData:
    c: np.ndarray
    integrality: np.ndarray
    bounds: Bounds
    constraint: LinearConstraint
    var: VarIndex
    big_m: float
    stages: list[int] | None = None


def _add_row(rows, lbs, ubs, coeffs: Iterable[tuple[int, float]], lb=-np.inf, ub=np.inf):
    rows.append(list(coeffs))
    lbs.append(lb)
    ubs.append(ub)


def build_stage_model(
    instance: FSTSPInstance,
    active_components: set[Component] | None = None,
    strengthen: bool = True,
    deadline: float | None = None,
    num_stages: int | None = None,
    fixed_truck_customers: list[int] | set[int] | None = None,
    fixed_drone_customers: list[int] | set[int] | None = None,
    fixed_truck_route: list[int] | None = None,
) -> ModelData:
    """Build the paper's 2-index stage-based MILP as a SciPy/HiGHS model.

    Component restrictions affect x, phi, A and B variables and are used by CMSA.
    When `active_components` is None, the full model is built.

    Four orthogonal fixing mechanisms are supported (paper Section 2 & 3):
    - `num_stages`: sets stage horizon K = {0, ..., num_stages - 1}.
    - `fixed_truck_customers`: forces specified customers to truck (phi_h = 0).
    - `fixed_drone_customers`: forces specified customers to drone (phi_h = 1, X_h^k = 0).
    - `fixed_truck_route`: fixes the exact sequence of locations visited by truck.
    """
    def check_deadline() -> None:
        if deadline is not None and time.perf_counter() >= deadline:
            raise ModelBuildTimeout("stage-model build deadline exceeded")

    check_deadline()
    V = instance.nodes
    C = instance.customers
    if fixed_truck_route is not None:
        K = list(range(len(fixed_truck_route)))
    elif num_stages is not None:
        K = list(range(num_stages))
    else:
        K = instance.stages
    S, E = instance.S, instance.E
    tau = instance.truck_time
    tau_d = instance.drone_time
    tL, tR, Dd = instance.launch_time, instance.recovery_time, instance.drone_endurance

    var = VarIndex()
    for k in K:
        check_deadline()
        for i in V:
            var.add(("X", k, i))
    for i in V:
        check_deadline()
        for j in V:
            var.add(("x", i, j))
    for h in C:
        var.add(("phi", h))
    for h in C:
        check_deadline()
        for i in V:
            var.add(("A", h, i))
            var.add(("B", h, i))
    for h in C:
        check_deadline()
        for k in K:
            var.add(("Y", h, k))
            var.add(("W", h, k))
    for k in K:
        check_deadline()
        for kp in K:
            if k < kp:
                var.add(("Z", k, kp))
    for i in V:
        var.add(("a", i))
        var.add(("d", i))

    nvar = var.size
    c = np.zeros(nvar)
    c[var[("d", E)]] = 1.0  # Eq. (1)
    lb = np.zeros(nvar)
    ub = np.full(nvar, np.inf)
    integrality = np.zeros(nvar, dtype=int)

    # Binary domains (Eqs. 41-52; Z stays continuous per Eq. 53).
    for key, idx in var._map.items():
        if key[0] in {"X", "x", "phi", "A", "B", "Y", "W"}:
            integrality[idx] = 1
            ub[idx] = 1.0
        elif key[0] == "Z":
            ub[idx] = 1.0

    # Tighten obvious infeasible variables from Eqs. (42)-(52) and self arcs.
    for i in V:
        if i != S:
            ub[var[("X", 0, i)]] = 0.0
    for k in K[1:]:
        ub[var[("X", k, S)]] = 0.0
    for i in V:
        if i != E:
            ub[var[("X", K[-1], i)]] = 0.0
    for i in V:
        ub[var[("x", i, i)]] = 0.0
    for h in C:
        ub[var[("A", h, E)]] = 0.0
        ub[var[("B", h, S)]] = 0.0
        ub[var[("Y", h, K[-1])]] = 0.0
        ub[var[("W", h, K[0])]] = 0.0
        if not instance.drone_allowed[h - 1]:
            ub[var[("phi", h)]] = 0.0

    # CMSA restricted-component fixing.
    if active_components is not None:
        for i in V:
            for j in V:
                if ("x", i, j) not in active_components:
                    ub[var[("x", i, j)]] = 0.0
        for h in C:
            if ("phi", h) not in active_components:
                ub[var[("phi", h)]] = 0.0
            for i in V:
                if ("A", h, i) not in active_components:
                    ub[var[("A", h, i)]] = 0.0
                if ("B", h, i) not in active_components:
                    ub[var[("B", h, i)]] = 0.0

    # 1. Fixed truck customers (customers that must be visited by truck: phi_h = 0)
    if fixed_truck_customers is not None:
        for h in fixed_truck_customers:
            ub[var[("phi", h)]] = 0.0
            for i_node in V:
                ub[var[("A", h, i_node)]] = 0.0
                ub[var[("B", h, i_node)]] = 0.0

    # 2. Fixed drone customers (customers that must be served by drone: phi_h = 1, X_h^k = 0, no truck arcs)
    if fixed_drone_customers is not None:
        for h in fixed_drone_customers:
            lb[var[("phi", h)]] = 1.0
            ub[var[("phi", h)]] = 1.0
            for k_idx in K:
                ub[var[("X", k_idx, h)]] = 0.0
            for i_node in V:
                ub[var[("x", i_node, h)]] = 0.0
                ub[var[("x", h, i_node)]] = 0.0

    # 3. Fixed truck route (exact sequence of locations visited by truck)
    if fixed_truck_route is not None:
        for k_idx in K:
            u_node = fixed_truck_route[k_idx]
            for i_node in V:
                if i_node == u_node:
                    lb[var[("X", k_idx, i_node)]] = 1.0
                    ub[var[("X", k_idx, i_node)]] = 1.0
                else:
                    ub[var[("X", k_idx, i_node)]] = 0.0
        route_edges = set(zip(fixed_truck_route[:-1], fixed_truck_route[1:]))
        for i_node in V:
            for j_node in V:
                if (i_node, j_node) in route_edges:
                    lb[var[("x", i_node, j_node)]] = 1.0
                    ub[var[("x", i_node, j_node)]] = 1.0
                else:
                    ub[var[("x", i_node, j_node)]] = 0.0
        truck_cust_set = set(fixed_truck_route[1:-1])
        for h in C:
            if h in truck_cust_set:
                ub[var[("phi", h)]] = 0.0
                for i_node in V:
                    ub[var[("A", h, i_node)]] = 0.0
                    ub[var[("B", h, i_node)]] = 0.0
            else:
                lb[var[("phi", h)]] = 1.0
                ub[var[("phi", h)]] = 1.0
                for i_node in V:
                    if i_node not in fixed_truck_route[:-1]:
                        ub[var[("A", h, i_node)]] = 0.0
                    if i_node not in fixed_truck_route[1:]:
                        ub[var[("B", h, i_node)]] = 0.0

    # A conservative Big-M for timing equations.
    max_leg = float(max(np.max(tau), np.max(tau_d), 1.0))
    M = (instance.n + 3) * (max_leg + tL + tR + Dd) * 2.0
    for i in V:
        ub[var[("a", i)]] = M
        ub[var[("d", i)]] = M

    rows: list[list[tuple[int, float]]] = []
    lbs: list[float] = []
    ubs: list[float] = []

    # Eqs. (2)-(5): depot conditions.
    _add_row(rows, lbs, ubs, [(var[("X", 0, S)], 1)], 1, 1)
    _add_row(rows, lbs, ubs, [(var[("X", k, S)], 1) for k in K[1:]], 0, 0)
    _add_row(rows, lbs, ubs, [(var[("X", k, E)], 1) for k in K[1:]], 1, 1)
    _add_row(rows, lbs, ubs, [(var[("x", E, i)], 1) for i in V], 0, 0)

    # Eqs. (6)-(9): truck routing and stage connectivity.
    for i in V:
        if i != E:
            coeff = [(var[("X", k, i)], 1) for k in K[:-1]] + [
                (var[("x", i, j)], -1) for j in V
            ]
            _add_row(rows, lbs, ubs, coeff, 0, 0)
        if i != S:
            coeff = [(var[("X", k, i)], 1) for k in K[1:]] + [
                (var[("x", j, i)], -1) for j in V
            ]
            _add_row(rows, lbs, ubs, coeff, 0, 0)
    for k in K:
        _add_row(rows, lbs, ubs, [(var[("X", k, i)], 1) for i in V], -np.inf, 1)
    for k in K[:-1]:
        check_deadline()
        for i in V:
            for j in V:
                # X[k,i] + x[i,j] - X[k+1,j] <= 1
                _add_row(
                    rows,
                    lbs,
                    ubs,
                    [(var[("X", k, i)], 1), (var[("x", i, j)], 1), (var[("X", k + 1, j)], -1)],
                    -np.inf,
                    1,
                )

    # Eqs. (10)-(11): consistency of launch/recovery locations and stages.
    for h in C:
        _add_row(
            rows, lbs, ubs,
            [(var[("A", h, i)], 1) for i in V] + [(var[("phi", h)], -1)], 0, 0,
        )
        _add_row(
            rows, lbs, ubs,
            [(var[("B", h, i)], 1) for i in V] + [(var[("phi", h)], -1)], 0, 0,
        )
        _add_row(
            rows, lbs, ubs,
            [(var[("Y", h, k)], 1) for k in K] + [(var[("phi", h)], -1)], 0, 0,
        )
        _add_row(
            rows, lbs, ubs,
            [(var[("W", h, k)], 1) for k in K] + [(var[("phi", h)], -1)], 0, 0,
        )

    # Eqs. (12)-(13): sortie-stage flow.
    for k in K:
        z_out = [(var[("Z", k, kp)], 1) for kp in K if k < kp]
        y = [(var[("Y", h, k)], -1) for h in C]
        if z_out or y:
            _add_row(rows, lbs, ubs, z_out + y, 0, 0)
        _add_row(rows, lbs, ubs, [(var[("Y", h, k)], 1) for h in C], -np.inf, 1)
        z_in = [(var[("Z", kp, k)], 1) for kp in K if kp < k]
        w = [(var[("W", h, k)], -1) for h in C]
        if z_in or w:
            _add_row(rows, lbs, ubs, z_in + w, 0, 0)
        _add_row(rows, lbs, ubs, [(var[("W", h, k)], 1) for h in C], -np.inf, 1)

    # Eq. (14): drone endurance.
    for h in C:
        coeff = [(var[("A", h, i)], float(tau_d[i, h])) for i in V]
        coeff += [(var[("B", h, i)], float(tau_d[h, i])) for i in V]
        coeff += [(var[("phi", h)], -(Dd - tR))]
        _add_row(rows, lbs, ubs, coeff, -np.inf, 0)

    # Eq. (15): no overlapping sorties.
    for m in K[:-1]:
        coeff = []
        for k in K:
            for kp in K:
                if k < kp and k <= m < kp:
                    coeff.append((var[("Z", k, kp)], 1))
        _add_row(rows, lbs, ubs, coeff, -np.inf, 1)

    # Eqs. (16)-(20): drone-truck forcing and customer assignment.
    for k in K:
        _add_row(
            rows, lbs, ubs,
            [(var[("Y", h, k)], 1) for h in C] + [(var[("X", k, i)], -1) for i in V],
            -np.inf, 0,
        )
        _add_row(
            rows, lbs, ubs,
            [(var[("W", h, k)], 1) for h in C] + [(var[("X", k, i)], -1) for i in V],
            -np.inf, 0,
        )
    for i in V:
        _add_row(
            rows, lbs, ubs,
            [(var[("A", h, i)], 1) for h in C] + [(var[("X", k, i)], -1) for k in K[:-1]],
            -np.inf, 0,
        )
        _add_row(
            rows, lbs, ubs,
            [(var[("B", h, i)], 1) for h in C] + [(var[("X", k, i)], -1) for k in K[1:]],
            -np.inf, 0,
        )
    for h in C:
        _add_row(
            rows, lbs, ubs,
            [(var[("X", k, h)], 1) for k in K] + [(var[("phi", h)], 1)], 1, 1,
        )

    # Eqs. (21)-(26): AND-style forcing between stage and location variables.
    for h in C:
        check_deadline()
        for i in V:
            for k in K:
                X = var[("X", k, i)]
                A, B = var[("A", h, i)], var[("B", h, i)]
                Y, W = var[("Y", h, k)], var[("W", h, k)]
                _add_row(rows, lbs, ubs, [(X, 1), (A, 1), (Y, -1)], -np.inf, 1)
                _add_row(rows, lbs, ubs, [(X, 1), (Y, 1), (A, -1)], -np.inf, 1)
                _add_row(rows, lbs, ubs, [(A, 1), (Y, 1), (X, -1)], -np.inf, 1)
                _add_row(rows, lbs, ubs, [(X, 1), (B, 1), (W, -1)], -np.inf, 1)
                _add_row(rows, lbs, ubs, [(X, 1), (W, 1), (B, -1)], -np.inf, 1)
                _add_row(rows, lbs, ubs, [(B, 1), (W, 1), (X, -1)], -np.inf, 1)

    # Eqs. (27)-(33): timing.
    _add_row(rows, lbs, ubs, [(var[("a", S)], 1)], 0, 0)
    _add_row(rows, lbs, ubs, [(var[("d", S)], 1)], 0, 0)
    for i in V:
        for j in V:
            if i == j:
                continue
            # Eq. 28: d_i - a_j + (tau_ij + M) x_ij <= M
            _add_row(
                rows, lbs, ubs,
                [(var[("d", i)], 1), (var[("a", j)], -1), (var[("x", i, j)], M + float(tau[i, j]))],
                -np.inf, M,
            )
    for h in C:
        for i in V:
            # Eq. 29
            _add_row(
                rows, lbs, ubs,
                [(var[("d", i)], 1), (var[("a", h)], -1), (var[("A", h, i)], M + float(tau_d[i, h]))],
                -np.inf, M,
            )
    for i in V:
        coeff = [(var[("a", i)], 1), (var[("d", i)], -1)]
        coeff += [(var[("A", h, i)], tL) for h in C]
        coeff += [(var[("B", h, i)], tR) for h in C]
        _add_row(rows, lbs, ubs, coeff, -np.inf, 0)
    for h in C:
        for j in V:
            coeff = [
                (var[("d", h)], 1),
                (var[("d", j)], -1),
                (var[("B", h, j)], M + float(tau_d[h, j]) + tR),
            ]
            coeff += [(var[("A", hp, j)], tL) for hp in C if hp != h]
            _add_row(rows, lbs, ubs, coeff, -np.inf, M)
    for h in C:
        check_deadline()
        for i in V:
            for j in V:
                coeff = [
                    (var[("d", i)], 1),
                    (var[("d", j)], -1),
                    (var[("A", h, i)], M + float(tau_d[i, h]) + 0.5 * tR),
                    (var[("B", h, j)], M + float(tau_d[h, j]) + 0.5 * tR),
                ]
                coeff += [(var[("A", hp, j)], tL) for hp in C]
                _add_row(rows, lbs, ubs, coeff, -np.inf, 2 * M)
                # Eq. 33
                q = M - 0.5 * (Dd - tR)
                _add_row(
                    rows, lbs, ubs,
                    [(var[("a", j)], 1), (var[("d", i)], -1), (var[("A", h, i)], q), (var[("B", h, j)], q)],
                    -np.inf, 2 * M,
                )

    if strengthen:
        # Eq. (34): final depot cannot occur in the first half of the stage horizon (paper Eq. 34).
        # We only forbid intermediate stages strictly before the final stage of K.
        first_forbidden = math.floor((instance.n + 2) / 2)
        for k in range(1, min(first_forbidden, len(K) - 1)):
            _add_row(rows, lbs, ubs, [(var[("X", k, E)], 1)], 0, 0)
        # Eq. (35): stage of final depot + number of drone customers = N + 2 (paper Eq. 35, where K = N + 2).
        # Constant K on RHS is ALWAYS instance.n + 2 (total original customers + 2 depots).
        coeff = [(var[("X", k, E)], k + 1) for k in K]
        coeff += [(var[("phi", h)], 1) for h in C]
        _add_row(rows, lbs, ubs, coeff, instance.n + 2, instance.n + 2)
        # Eqs. (36)-(37): objective lower bounds.
        coeff = [(var[("x", i, j)], float(tau[i, j])) for i in V for j in V]
        coeff += [(var[("A", h, i)], tL) for h in C for i in V]
        coeff += [(var[("B", h, i)], tR) for h in C for i in V]
        coeff += [(var[("d", E)], -1)]
        _add_row(rows, lbs, ubs, coeff, -np.inf, 0)
        coeff = [(var[("A", h, i)], float(tau_d[i, h]) + tL) for h in C for i in V]
        coeff += [(var[("B", h, j)], float(tau_d[h, j]) + tR) for h in C for j in V]
        coeff += [(var[("d", E)], -1)]
        _add_row(rows, lbs, ubs, coeff, -np.inf, 0)
        # Eq. (38): recovery follows launch.
        for h in C:
            for k in K[:-1]:
                coeff = [(var[("W", h, p)], 1) for p in range(0, k + 2)]
                coeff += [(var[("Y", h, p)], -1) for p in range(0, k + 1)]
                _add_row(rows, lbs, ubs, coeff, -np.inf, 0)
        # Eqs. (39)-(40): launch and recovery cannot be same location/stage.
        for h in C:
            for i in V:
                _add_row(
                    rows, lbs, ubs,
                    [(var[("A", h, i)], 1), (var[("B", h, i)], 1), (var[("phi", h)], -1)],
                    -np.inf, 0,
                )
            for k in K:
                _add_row(
                    rows, lbs, ubs,
                    [(var[("Y", h, k)], 1), (var[("W", h, k)], 1), (var[("phi", h)], -1)],
                    -np.inf, 0,
                )

    check_deadline()
    A_mat = lil_matrix((len(rows), nvar), dtype=float)
    for r, coeffs in enumerate(rows):
        if r % 2048 == 0:
            check_deadline()
        for idx, value in coeffs:
            A_mat[r, idx] += value
    constraint = LinearConstraint(A_mat.tocsr(), np.asarray(lbs), np.asarray(ubs))
    return ModelData(c, integrality, Bounds(lb, ub), constraint, var, M, stages=K)


def extract_model_metrics(model: ModelData, active_components: set[Component] | None = None) -> dict:
    """Extract separated pre-presolve and structural metrics for Table 4 audit compliance."""
    lb = np.asarray(model.bounds.lb)
    ub = np.asarray(model.bounds.ub)
    nvar = model.var.size if model.var is not None else (model.c.size if hasattr(model.c, "size") else len(model.c))

    free_var_mask = (ub > lb + 1e-9)
    fixed_zero_mask = (np.abs(lb) < 1e-9) & (np.abs(ub) < 1e-9)
    fixed_one_mask = (np.abs(lb - 1.0) < 1e-9) & (np.abs(ub - 1.0) < 1e-9)

    free_variables = int(np.sum(free_var_mask))
    fixed_zero_variables = int(np.sum(fixed_zero_mask))
    fixed_one_variables = int(np.sum(fixed_one_mask))

    A_csr = model.constraint.A.tocsr()
    A_free = A_csr[:, free_var_mask]
    row_free_nnz = np.diff(A_free.indptr)
    active_constraints_before_presolve = int(np.sum(row_free_nnz > 0))
    active_nonzeros_before_presolve = int(A_free.nnz)

    return {
        "original_variables": nvar,
        "original_constraints": int(A_csr.shape[0]),
        "original_nonzeros": int(getattr(A_csr, "nnz", 0)),
        "fixed_zero_variables": fixed_zero_variables,
        "fixed_one_variables": fixed_one_variables,
        "free_variables": free_variables,
        "active_constraints_before_presolve": active_constraints_before_presolve,
        "active_nonzeros_before_presolve": active_nonzeros_before_presolve,
        "presolved_variables": None,
        "presolved_constraints": None,
        "presolved_nonzeros": None,
        "presolve_metrics_available": False,
        "presolve_metrics_note": (
            "HiGHS open-source solver wrapper in scipy.optimize.milp does not expose post-presolve "
            "dimensions; presolved metrics are null. Active pre-presolve metrics reflect the compact subproblem."
        ),
        # Backward-compatibility aliases:
        "n_variables": nvar,
        "n_constraints": int(A_csr.shape[0]),
        "n_nonzeros": int(getattr(A_csr, "nnz", 0)),
        "n_active_variables": free_variables,
        "n_active_constraints": active_constraints_before_presolve,
        "n_active_nonzeros": active_nonzeros_before_presolve,
        "n_fixed_zero_variables": fixed_zero_variables,
        "n_fixed_one_variables": fixed_one_variables,
        "n_free_variables": free_variables,
        "n_active_components": len(active_components) if active_components is not None else None,
    }


def solve_stage_model(
    instance: FSTSPInstance,
    time_limit: float = 60.0,
    mip_rel_gap: float = 0.0,
    active_components: set[Component] | None = None,
    strengthen: bool = True,
    deadline: float | None = None,
    num_stages: int | None = None,
    fixed_truck_customers: list[int] | set[int] | None = None,
    fixed_drone_customers: list[int] | set[int] | None = None,
    fixed_truck_route: list[int] | None = None,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> FSTSPSolution:
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if not 0 <= mip_rel_gap < 1:
        raise ValueError("mip_rel_gap must be in [0, 1)")

    from fstsp.solver import get_solver_backend, SolverOptions
    backend = get_solver_backend(solver_backend)

    started = time.perf_counter()
    try:
        model = build_stage_model(
            instance,
            active_components=active_components,
            strengthen=strengthen,
            deadline=deadline,
            num_stages=num_stages,
            fixed_truck_customers=fixed_truck_customers,
            fixed_drone_customers=fixed_drone_customers,
            fixed_truck_route=fixed_truck_route,
        )
    except ModelBuildTimeout:
        return FSTSPSolution(
            feasible=False, objective=None, runtime=time.perf_counter() - started,
            status="model_build_deadline_exceeded",
            metadata={"solver": solver_backend, "solver_backend": solver_backend},
        )
    remaining_limit = float(time_limit)
    if deadline is not None:
        remaining_limit = min(remaining_limit, max(0.0, deadline - time.perf_counter()))
        if remaining_limit <= 1e-4:
            return FSTSPSolution(
                feasible=False, objective=None, runtime=time.perf_counter() - started,
                status="deadline_exhausted_before_mip",
                metadata={"solver": solver_backend, "solver_backend": solver_backend},
            )

    options = SolverOptions(
        time_limit=remaining_limit,
        mip_rel_gap=float(mip_rel_gap),
        threads=threads,
        mip_emphasis=mip_emphasis,
        presolve=True,
        deadline=deadline,
    )
    result = backend.solve(model, options)
    runtime = time.perf_counter() - started
    model_stats = extract_model_metrics(model, active_components)
    if result.model_metrics:
        model_stats.update(result.model_metrics)

    if result.x is None:
        return FSTSPSolution(
            feasible=False,
            objective=None,
            runtime=runtime,
            status=str(result.status),
            metadata={
                "solver": result.solver_name,
                "solver_backend": solver_backend,
                "solver_version": result.solver_version,
                "status_code": result.raw_status,
                **model_stats,
            },
        )
    xval = np.asarray(result.x, dtype=float)
    if xval.shape != (model.var.size,) or not np.all(np.isfinite(xval)):
        return FSTSPSolution(
            feasible=False,
            objective=None,
            runtime=runtime,
            status="solver_returned_invalid_vector",
            metadata={
                "solver": result.solver_name,
                "solver_backend": solver_backend,
                "solver_version": result.solver_version,
                "status_code": result.raw_status,
                **model_stats,
            },
        )

    # Do not trust the backend status blindly. Check the returned incumbent against
    # the exact bounds, linear rows, and integer domains used to build the model.
    # This catches numerical/tolerance regressions before discrete extraction.
    lower = np.asarray(model.bounds.lb, dtype=float)
    upper = np.asarray(model.bounds.ub, dtype=float)
    bound_violation = float(
        max(
            np.max(np.maximum(lower - xval, 0.0), initial=0.0),
            np.max(np.maximum(xval - upper, 0.0), initial=0.0),
        )
    )
    row_value = np.asarray(model.constraint.A @ xval, dtype=float)
    row_lb = np.asarray(model.constraint.lb, dtype=float)
    row_ub = np.asarray(model.constraint.ub, dtype=float)
    row_violation = float(
        max(
            np.max(np.maximum(row_lb - row_value, 0.0), initial=0.0),
            np.max(np.maximum(row_value - row_ub, 0.0), initial=0.0),
        )
    )
    integer_mask = model.integrality != 0
    integrality_violation = float(
        np.max(np.abs(xval[integer_mask] - np.rint(xval[integer_mask])), initial=0.0)
    )
    numerical_tol = 2e-5
    vector_diag = {
        "max_bound_violation": bound_violation,
        "max_row_violation": row_violation,
        "max_integrality_violation": integrality_violation,
        "vector_validation_tolerance": numerical_tol,
    }
    if max(bound_violation, row_violation, integrality_violation) > numerical_tol:
        return FSTSPSolution(
            feasible=False,
            objective=None,
            runtime=runtime,
            status="solver_incumbent_failed_model_residual_check",
            metadata={
                "solver": result.solver_name,
                "solver_backend": solver_backend,
                "solver_version": result.solver_version,
                "status_code": result.raw_status,
                **model_stats,
                **vector_diag,
            },
        )

    stages = getattr(model, "stages", None) or instance.stages
    route_by_stage: list[tuple[int, int]] = []
    for k in stages:
        chosen = [i for i in instance.nodes if xval[model.var[("X", k, i)]] > 0.5]
        if len(chosen) > 1:
            return FSTSPSolution(
                feasible=False, objective=None, runtime=runtime,
                status="ambiguous_truck_stage_extraction",
                metadata={"stage": k, "chosen": chosen, **vector_diag},
            )
        if chosen:
            route_by_stage.append((k, chosen[0]))
    route = [node for _, node in route_by_stage]

    sorties: list[DroneSortie] = []
    for h in instance.customers:
        if xval[model.var[("phi", h)]] > 0.5:
            launches = [i for i in instance.nodes if xval[model.var[("A", h, i)]] > 0.5]
            recoveries = [i for i in instance.nodes if xval[model.var[("B", h, i)]] > 0.5]
            launch_stages = [k for k in stages if xval[model.var[("Y", h, k)]] > 0.5]
            recovery_stages = [k for k in stages if xval[model.var[("W", h, k)]] > 0.5]
            if not (len(launches) == len(recoveries) == len(launch_stages) == len(recovery_stages) == 1):
                return FSTSPSolution(
                    feasible=False, objective=None, runtime=runtime,
                    status="ambiguous_drone_sortie_extraction",
                    metadata={
                        "customer": h,
                        "launches": launches,
                        "recoveries": recoveries,
                        "launch_stages": launch_stages,
                        "recovery_stages": recovery_stages,
                        **vector_diag,
                    },
                )
            sorties.append(
                DroneSortie(
                    launches[0], h, recoveries[0], launch_stages[0], recovery_stages[0]
                )
            )
    gap = result.mip_gap
    solver_objective = float(result.objective) if result.objective is not None else float("nan")
    if not math.isfinite(solver_objective):
        return FSTSPSolution(
            feasible=False,
            objective=None,
            runtime=runtime,
            status="solver_returned_nonfinite_objective",
            metadata={
                "solver": result.solver_name,
                "solver_backend": solver_backend,
                "solver_version": result.solver_version,
                "status_code": result.raw_status,
            },
        )

    if result.solver_name == "cplex":
        proven_optimal = bool(
            (result.raw_status in (101, 102))
            or (gap is not None and gap <= 1e-9)
        )
    else:
        proven_optimal = bool(result.raw_status == 0 and (gap is None or gap <= 1e-9))

    candidate = FSTSPSolution(
        feasible=True,
        objective=solver_objective,
        truck_route=route,
        drone_sorties=sorties,
        runtime=runtime,
        mip_gap=gap,
        status=str(result.status),
        metadata={
            "solver": result.solver_name,
            "solver_backend": solver_backend,
            "solver_version": result.solver_version,
            "status_code": result.raw_status,
            "best_bound": result.best_bound,
            "node_count": result.node_count,
            **model_stats,
            "big_m": model.big_m,
            "solver_objective": solver_objective,
            "requested_mip_rel_gap": float(mip_rel_gap),
            "proven_optimal": proven_optimal,
            **vector_diag,
        },
    )

    # Independently reconstruct the physical schedule from the discrete solution.
    # This catches extraction/tolerance/modeling regressions before a result is
    # reported as feasible. We report the certified completion time, retaining the
    # raw solver objective in metadata for auditability.
    from fstsp.evaluation.schedule import evaluate_schedule

    certification = evaluate_schedule(instance, candidate)
    if not certification.feasible or certification.completion_time is None:
        candidate.feasible = False
        candidate.objective = None
        candidate.status = "solver_incumbent_failed_schedule_certification"
        candidate.metadata["certification_issues"] = certification.issues
        return candidate

    objective_delta = certification.completion_time - solver_objective
    objective_tol = 1e-5 + 1e-7 * max(
        1.0, abs(certification.completion_time), abs(solver_objective)
    )
    candidate.metadata["certified_objective"] = certification.completion_time
    candidate.metadata["objective_delta_vs_solver"] = objective_delta
    candidate.metadata["objective_certification_tolerance"] = objective_tol
    if abs(objective_delta) > objective_tol:
        candidate.feasible = False
        candidate.objective = None
        candidate.status = "solver_objective_failed_schedule_certification"
        candidate.metadata["certification_issues"] = [
            "raw solver objective and reconstructed completion time differ beyond tolerance"
        ]
        return candidate

    candidate.objective = certification.completion_time
    return candidate


def solution_components(solution: FSTSPSolution) -> set[Component]:
    comp: set[Component] = set()
    if not solution.feasible:
        return comp
    for i, j in zip(solution.truck_route, solution.truck_route[1:]):
        comp.add(("x", i, j))
    for s in solution.drone_sorties:
        comp.add(("phi", s.customer))
        comp.add(("A", s.customer, s.launch_node))
        comp.add(("B", s.customer, s.recovery_node))
    return comp
