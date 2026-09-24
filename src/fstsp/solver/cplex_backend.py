from __future__ import annotations

import math
import time
import numpy as np

from fstsp.formulation.stage_based import ModelData, extract_model_metrics
from fstsp.solver.base import SolverBackend, SolverOptions, SolverResult


class CplexNotAvailableError(RuntimeError):
    """Raised when CPLEX backend is requested but IBM ILOG CPLEX is not installed."""
    pass


class CplexBackend(SolverBackend):
    """Full IBM ILOG CPLEX solver backend implementing official paper experimental protocol.

    Paper Section 4 configuration:
    - IBM ILOG CPLEX 22.11
    - MIP Emphasis = 5 (Feasibility priority)
    - Threads = 8
    - Time limit = 15s (restricted MIP) or 3600s/7200s (exact)
    """

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cplex  # noqa: F401
            return True
        except ImportError:
            return False

    def solve(self, model: ModelData, options: SolverOptions) -> SolverResult:
        if not self.is_available():
            raise CplexNotAvailableError(
                "CPLEX backend was requested per paper protocol, but IBM ILOG CPLEX ('cplex' Python package) "
                "is not installed or licensed in this environment. In paper reproduction mode, "
                "we do NOT silently fall back to HiGHS. Please install CPLEX Studio 22.11 or "
                "explicitly configure solver backend to 'highs' for the open-source alternative."
            )

        import cplex
        from cplex.exceptions import CplexError

        started = time.perf_counter()
        c = cplex.Cplex()

        # Problem setup
        c.set_problem_name("FSTSP_Stage_Based_MILP")
        c.objective.set_sense(c.objective.sense.minimize)

        # Variables: obj, lb, ub, types, names
        nvar = model.var.size
        obj = model.c.tolist()
        lb = [0.0 if np.isneginf(x) else float(x) for x in model.bounds.lb]
        ub = [1e20 if np.isposinf(x) else float(x) for x in model.bounds.ub]
        types = [
            c.variables.type.binary if int(model.integrality[i]) == 1 and ub[i] <= 1.0 + 1e-9
            else c.variables.type.integer if int(model.integrality[i]) == 1
            else c.variables.type.continuous
            for i in range(nvar)
        ]
        var_names = [f"v_{i}" for i in range(nvar)]
        c.variables.add(obj=obj, lb=lb, ub=ub, types=types, names=var_names)

        # Linear constraints from CSR matrix
        A_csr = model.constraint.A.tocsr()
        lhs = np.asarray(model.constraint.lb)
        rhs = np.asarray(model.constraint.ub)
        nrows = A_csr.shape[0]

        lin_expr = []
        senses = []
        rhs_vals = []

        for r in range(nrows):
            start_idx = A_csr.indptr[r]
            end_idx = A_csr.indptr[r + 1]
            cols = A_csr.indices[start_idx:end_idx].tolist()
            vals = A_csr.data[start_idx:end_idx].tolist()
            pair = cplex.SparsePair(ind=cols, val=vals)

            lo = float(lhs[r])
            hi = float(rhs[r])

            if abs(lo - hi) < 1e-9:
                # Equality constraint
                lin_expr.append(pair)
                senses.append("E")
                rhs_vals.append(hi)
            elif np.isneginf(lo) and math.isfinite(hi):
                # Less-than-or-equal constraint
                lin_expr.append(pair)
                senses.append("L")
                rhs_vals.append(hi)
            elif math.isfinite(lo) and np.isposinf(hi):
                # Greater-than-or-equal constraint
                lin_expr.append(pair)
                senses.append("G")
                rhs_vals.append(lo)
            elif math.isfinite(lo) and math.isfinite(hi):
                # Range constraint: add two rows (G for lo, L for hi)
                lin_expr.append(pair)
                senses.append("G")
                rhs_vals.append(lo)
                lin_expr.append(pair)
                senses.append("L")
                rhs_vals.append(hi)

        c.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs_vals)

        # Configure CPLEX parameters per paper protocol
        remaining_time = float(options.time_limit)
        if options.deadline is not None:
            remaining_time = min(remaining_time, max(0.0, options.deadline - time.perf_counter()))
            if remaining_time <= 1e-4:
                return SolverResult(
                    status="deadline_exhausted_before_mip",
                    raw_status=-1,
                    success=False,
                    x=None,
                    objective=None,
                    best_bound=None,
                    mip_gap=None,
                    runtime=time.perf_counter() - started,
                    solver_name="cplex",
                    solver_version=cplex.__version__,
                    model_metrics=extract_model_metrics(model),
                    message="Deadline exhausted before starting CPLEX solve",
                )

        c.parameters.timelimit.set(float(remaining_time))
        if options.threads is not None and options.threads > 0:
            c.parameters.threads.set(int(options.threads))
        if options.mip_emphasis is not None:
            c.parameters.emphasis.mip.set(int(options.mip_emphasis))
        if options.mip_rel_gap > 0:
            c.parameters.mip.tolerances.mipgap.set(float(options.mip_rel_gap))
        if not options.presolve:
            c.parameters.preprocessing.presolve.set(0)

        # Solve
        c.solve()
        runtime = time.perf_counter() - started

        # Extract CPLEX statistics for Table 4 compliance
        reported_vars = c.variables.get_num()
        reported_cons = c.linear_constraints.get_num()
        reported_nnz = c.linear_constraints.get_num_nonzeros()

        # Query presolved dimensions if available
        presolved_vars = None
        presolved_cons = None
        presolved_nnz = None
        try:
            # Query post-presolve dimensions from CPLEX internal statistics
            p_vars = c.solution.progress.get_num_variables()
            p_cons = c.solution.progress.get_num_constraints()
            if p_vars is not None and p_vars > 0:
                presolved_vars = int(p_vars)
                presolved_cons = int(p_cons) if p_cons is not None else None
        except (AttributeError, CplexError):
            pass

        metrics = extract_model_metrics(model)
        metrics.update(
            {
                "solver_reported_variables": reported_vars,
                "solver_reported_constraints": reported_cons,
                "solver_reported_nonzeros": reported_nnz,
                "presolved_variables": presolved_vars,
                "presolved_constraints": presolved_cons,
                "presolved_nonzeros": presolved_nnz,
                "presolve_metrics_available": presolved_vars is not None,
                "presolve_metrics_note": "Reported directly by IBM ILOG CPLEX",
            }
        )

        status_code = c.solution.get_status()
        status_str = c.solution.get_status_string()

        # Check for solution vector
        xval = None
        obj = None
        best_bound = None
        mip_gap = None
        node_count = None

        try:
            xval = np.asarray(c.solution.get_values(), dtype=float)
            obj = float(c.solution.get_objective_value())
        except CplexError:
            xval = None
            obj = None

        try:
            best_bound = float(c.solution.MIP.get_best_objective())
        except (CplexError, AttributeError):
            best_bound = None

        try:
            mip_gap = float(c.solution.MIP.get_mip_relative_gap())
        except (CplexError, AttributeError):
            mip_gap = None

        try:
            node_count = int(c.solution.progress.get_num_nodes_processed())
        except (CplexError, AttributeError):
            node_count = None

        # Determine success
        # Status 101: MIP_optimal, 102: MIP_optimal_tol, 107: MIP_time_lim_feasible, etc.
        feasible_statuses = {101, 102, 104, 105, 107, 108, 111, 112, 113, 114}
        success = (status_code in feasible_statuses) and (xval is not None)

        return SolverResult(
            status=status_str,
            raw_status=status_code,
            success=success,
            x=xval,
            objective=obj,
            best_bound=best_bound,
            mip_gap=mip_gap,
            runtime=runtime,
            node_count=node_count,
            solver_name="cplex",
            solver_version=cplex.__version__,
            model_metrics=metrics,
            message=status_str,
        )
