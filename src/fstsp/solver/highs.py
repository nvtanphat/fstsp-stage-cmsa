from __future__ import annotations

import math
import time
import numpy as np
from scipy.optimize import milp

from fstsp.formulation.stage_based import ModelData, extract_model_metrics
from fstsp.solver.base import SolverBackend, SolverOptions, SolverResult


class HighsBackend(SolverBackend):
    """Open-source HiGHS solver backend using scipy.optimize.milp."""

    @classmethod
    def is_available(cls) -> bool:
        return True

    def solve(self, model: ModelData, options: SolverOptions) -> SolverResult:
        started = time.perf_counter()
        remaining_limit = float(options.time_limit)
        if options.deadline is not None:
            remaining_limit = min(remaining_limit, max(0.0, options.deadline - time.perf_counter()))
            if remaining_limit <= 1e-4:
                return SolverResult(
                    status="deadline_exhausted_before_mip",
                    raw_status=-1,
                    success=False,
                    x=None,
                    objective=None,
                    best_bound=None,
                    mip_gap=None,
                    runtime=time.perf_counter() - started,
                    solver_name="highs",
                    solver_version="scipy-milp",
                    model_metrics=extract_model_metrics(model),
                    message="Deadline exhausted before starting MIP solve",
                )

        milp_options = {
            "time_limit": float(remaining_limit),
            "mip_rel_gap": float(options.mip_rel_gap),
            "presolve": bool(options.presolve),
        }

        res = milp(
            c=model.c,
            integrality=model.integrality,
            bounds=model.bounds,
            constraints=model.constraint,
            options=milp_options,
        )
        runtime = time.perf_counter() - started
        metrics = extract_model_metrics(model)

        raw_status = int(res.status)
        msg = str(res.message)
        gap = getattr(res, "mip_gap", None)
        if gap is not None:
            try:
                gap = float(gap)
            except (TypeError, ValueError):
                gap = None
            if gap is not None and not math.isfinite(gap):
                gap = None

        best_bound = getattr(res, "mip_dual_bound", None)
        if best_bound is not None:
            try:
                best_bound = float(best_bound)
            except (TypeError, ValueError):
                best_bound = None
            if best_bound is not None and not math.isfinite(best_bound):
                best_bound = None

        node_count = getattr(res, "mip_node_count", None)
        if node_count is not None:
            try:
                node_count = int(node_count)
            except (TypeError, ValueError):
                node_count = None

        if res.x is None:
            return SolverResult(
                status=msg,
                raw_status=raw_status,
                success=False,
                x=None,
                objective=None,
                best_bound=best_bound,
                mip_gap=gap,
                runtime=runtime,
                node_count=node_count,
                solver_name="highs",
                solver_version="scipy-milp",
                model_metrics=metrics,
                message=msg,
            )

        xval = np.asarray(res.x, dtype=float)
        obj = float(res.fun) if res.fun is not None and math.isfinite(res.fun) else None

        return SolverResult(
            status=msg,
            raw_status=raw_status,
            success=bool(res.success),
            x=xval,
            objective=obj,
            best_bound=best_bound,
            mip_gap=gap,
            runtime=runtime,
            node_count=node_count,
            solver_name="highs",
            solver_version="scipy-milp",
            model_metrics=metrics,
            message=msg,
        )
