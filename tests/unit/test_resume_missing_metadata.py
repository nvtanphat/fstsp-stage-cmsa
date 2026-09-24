from __future__ import annotations

import json
from pathlib import Path
import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.solution import FSTSPSolution
import kaggle.run_experiment as kre


def test_resume_rejects_missing_metadata(tmp_path):
    """Verify that cached solutions with missing metadata are marked LEGACY_UNVERIFIED and rejected."""
    inst = generate_uniform_instance(n=6, seed=1)
    tag = "test_legacy_unverified_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    # 1. Solution missing instance_hash and schema_version
    sol_legacy = FSTSPSolution(
        feasible=True,
        objective=150.0,
        runtime=1800.0,
        truck_route=[0, 1, 2, 3, 4, 5, 7],
        drone_sorties=[],
        status="cmsa_completed",
        metadata={
            "solver_backend": "highs",
            "solver_seed": 1,
            "method": "cmsa",
        },
    )
    sol_legacy.to_json(sol_dir / "solution.json")

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        settings = {
            "solver_backend": "highs",
            "solver_seed": 1,
            "total_time": 1800.0,
            "mip_time": 15.0,
            "age_limit": 2,
            "resume": True,
        }
        res = kre._find_cached_solution(
            tag,
            min_budget=1800.0,
            expected_instance_hash=kre.compute_instance_hash(inst),
            expected_backend="highs",
            expected_seed=1,
            expected_age_limit=2,
            expected_method="cmsa",
            inst=inst,
            settings=settings,
        )
        assert res is None, "Legacy cache without instance_hash must be rejected!"
    finally:
        kre.OUT = orig_out
