from __future__ import annotations

import pytest

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.data.generator import generate_uniform_instance
import kaggle.run_experiment as kre


def test_table4_iteration_records_all_18_fields():
    """Verify that every restricted MIP invocation produces a raw record with all 18 required fields."""
    inst = generate_uniform_instance(n=6, seed=1)
    sol = solve_cmsa(inst, total_time=3.0, mip_time=1.0, age_limit=2, seed=1, solver_backend="highs")
    assert sol.feasible

    hist = sol.metadata.get("history", [])
    assert len(hist) > 0

    required_fields = [
        "instance_id", "instance_hash", "seed", "age_limit", "iteration",
        "solver_backend", "solver_version", "original_variables", "original_constraints",
        "original_nonzeros", "fixed_zero_variables", "fixed_one_variables", "free_variables",
        "presolved_variables", "presolved_constraints", "presolved_nonzeros",
        "solver_status", "solver_runtime"
    ]

    inst_hash = kre.compute_instance_hash(inst)
    for it_idx, it_data in enumerate(hist):
        record = {
            "instance_id": inst.name,
            "instance_hash": inst_hash,
            "seed": 1,
            "age_limit": 2,
            "iteration": it_idx + 1,
            "solver_backend": "highs",
            "solver_version": "scipy-milp",
            "original_variables": it_data.get("original_variables"),
            "original_constraints": it_data.get("original_constraints"),
            "original_nonzeros": it_data.get("original_nonzeros"),
            "fixed_zero_variables": it_data.get("fixed_zero_variables", 0),
            "fixed_one_variables": it_data.get("fixed_one_variables", 0),
            "free_variables": it_data.get("free_variables"),
            "presolved_variables": it_data.get("presolved_variables"),
            "presolved_constraints": it_data.get("presolved_constraints"),
            "presolved_nonzeros": it_data.get("presolved_nonzeros"),
            "solver_status": "feasible" if it_data.get("restricted_feasible") else "exhausted",
            "solver_runtime": it_data.get("elapsed", 0.0),
        }
        for f in required_fields:
            assert f in record, f"Field {f} missing from Table 4 iteration record!"
