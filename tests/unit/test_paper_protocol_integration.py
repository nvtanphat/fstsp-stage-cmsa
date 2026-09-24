from __future__ import annotations

from pathlib import Path
import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution
import kaggle.run_experiment as kre


def test_end_to_end_pipeline_and_config_change_rejection(tmp_path):
    """End-to-end test verifying execution pipeline and cache invalidation on config change."""
    inst = generate_uniform_instance(n=4, seed=1)
    tag = "test_e2e_tag"

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        settings_base = {
            "sizes": [4],
            "seeds": [1],
            "total_time": 2.0,
            "mip_time": 1.0,
            "age_limit": 2,
            "solver_backend": "highs",
            "solver_seed": 1,
            "resume": True,
        }

        # 1. Run fresh instance through CMSA pipeline
        res1 = kre._solve_and_record_cmsa(inst, tag, settings_base)
        assert res1["feasible"] is True
        assert res1["objective"] is not None
        assert res1["resumed"] is False

        # 2. Check solution file and validator
        sol_file = tmp_path / tag / "solution.json"
        assert sol_file.is_file()

        # 3. Resume with identical configuration -> should succeed
        res2 = kre._solve_and_record_cmsa(inst, tag, settings_base)
        assert res2["resumed"] is True
        assert res2["objective"] == res1["objective"]

        # 4. Change configuration (e.g. solver_seed changed to 2) -> must reject cache!
        settings_modified = {
            **settings_base,
            "solver_seed": 2,
        }
        res3 = kre._solve_and_record_cmsa(inst, tag, settings_modified)
        assert res3["resumed"] is False, "Modified solver_seed must reject old cache!"
    finally:
        kre.OUT = orig_out
