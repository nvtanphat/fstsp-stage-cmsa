from __future__ import annotations

import json
from pathlib import Path
import pytest

from fstsp.domain.solution import FSTSPSolution
import kaggle.run_experiment as kre


def test_resume_provenance_validation(tmp_path):
    """Verify _find_cached_solution strictly checks instance_hash, backend, and age_limit."""
    tag = "test_provenance_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    sol = FSTSPSolution(
        feasible=True,
        objective=150.0,
        runtime=100.0,
        truck_route=[0, 1, 2, 3],
        drone_sorties=[],
        status="feasible",
        metadata={
            "instance_hash": "hash_aaa",
            "solver_backend": "highs",
            "age_limit": 2,
        },
    )
    sol.to_json(sol_dir / "solution.json")

    # Monkeypatch search dir
    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        # 1. Matching hashes -> successfully found
        match = kre._find_cached_solution(
            tag,
            expected_instance_hash="hash_aaa",
            expected_backend="highs",
            expected_age_limit=2,
        )
        assert match is not None
        assert match[0].objective == 150.0

        # 2. Mismatched instance_hash -> rejected
        mismatch_inst = kre._find_cached_solution(
            tag,
            expected_instance_hash="hash_bbb",
            expected_backend="highs",
            expected_age_limit=2,
        )
        assert mismatch_inst is None

        # 3. Mismatched solver_backend -> rejected
        mismatch_backend = kre._find_cached_solution(
            tag,
            expected_instance_hash="hash_aaa",
            expected_backend="cplex",
            expected_age_limit=2,
        )
        assert mismatch_backend is None

        # 4. Mismatched age_limit -> rejected
        mismatch_age = kre._find_cached_solution(
            tag,
            expected_instance_hash="hash_aaa",
            expected_backend="highs",
            expected_age_limit=5,
        )
        assert mismatch_age is None
    finally:
        kre.OUT = orig_out
