from __future__ import annotations

from pathlib import Path
import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.solution import FSTSPSolution
import kaggle.run_experiment as kre


def test_resume_rejects_backend_mismatch(tmp_path):
    """Verify that cached solutions from HiGHS are rejected when CPLEX is requested."""
    inst = generate_uniform_instance(n=6, seed=1)
    tag = "test_backend_mismatch_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    sol = FSTSPSolution(
        feasible=True,
        objective=150.0,
        runtime=1800.0,
        truck_route=[0, 1, 2, 3, 4, 5, 7],
        drone_sorties=[],
        status="cmsa_completed",
        metadata={
            "schema_version": kre.SCHEMA_VERSION,
            "instance_hash": kre.compute_instance_hash(inst),
            "solver_backend": "highs",
            "solver_seed": 1,
            "method": "cmsa",
            "total_time": 1800.0,
            "mip_time": 15.0,
            "age_limit": 2,
        },
    )
    sol.to_json(sol_dir / "solution.json")

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        # Request CPLEX backend
        res = kre._find_cached_solution(
            tag,
            min_budget=1800.0,
            expected_instance_hash=kre.compute_instance_hash(inst),
            expected_backend="cplex",
            expected_seed=1,
            expected_age_limit=2,
            expected_method="cmsa",
            inst=inst,
        )
        assert res is None, "Cache with highs must be rejected when cplex is requested!"
    finally:
        kre.OUT = orig_out


def test_resume_rejects_budget_mismatch(tmp_path):
    """Verify that cached solutions from 45s smoke test are rejected when 1800s paper protocol is requested."""
    inst = generate_uniform_instance(n=6, seed=1)
    tag = "test_budget_mismatch_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    sol = FSTSPSolution(
        feasible=True,
        objective=150.0,
        runtime=45.0,
        truck_route=[0, 1, 2, 3, 4, 5, 7],
        drone_sorties=[],
        status="cmsa_completed",
        metadata={
            "schema_version": kre.SCHEMA_VERSION,
            "instance_hash": kre.compute_instance_hash(inst),
            "solver_backend": "cplex",
            "solver_seed": 1,
            "method": "cmsa",
            "total_time": 45.0,
            "mip_time": 8.0,
            "age_limit": 2,
        },
    )
    sol.to_json(sol_dir / "solution.json")

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        res = kre._find_cached_solution(
            tag,
            min_budget=1800.0,
            expected_instance_hash=kre.compute_instance_hash(inst),
            expected_backend="cplex",
            expected_seed=1,
            expected_age_limit=2,
            expected_method="cmsa",
            inst=inst,
        )
        assert res is None, "45s cache must be rejected when 1800s is required!"
    finally:
        kre.OUT = orig_out


def test_resume_rejects_age_limit_mismatch(tmp_path):
    """Verify that cached solutions with age=5 are rejected when age=2 is requested."""
    inst = generate_uniform_instance(n=6, seed=1)
    tag = "test_age_mismatch_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    sol = FSTSPSolution(
        feasible=True,
        objective=150.0,
        runtime=1800.0,
        truck_route=[0, 1, 2, 3, 4, 5, 7],
        drone_sorties=[],
        status="cmsa_completed",
        metadata={
            "schema_version": kre.SCHEMA_VERSION,
            "instance_hash": kre.compute_instance_hash(inst),
            "solver_backend": "cplex",
            "solver_seed": 1,
            "method": "cmsa",
            "total_time": 1800.0,
            "mip_time": 15.0,
            "age_limit": 5,
        },
    )
    sol.to_json(sol_dir / "solution.json")

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        res = kre._find_cached_solution(
            tag,
            min_budget=1800.0,
            expected_instance_hash=kre.compute_instance_hash(inst),
            expected_backend="cplex",
            expected_seed=1,
            expected_age_limit=2,
            expected_method="cmsa",
            inst=inst,
        )
        assert res is None, "age=5 cache must be rejected when age=2 is requested!"
    finally:
        kre.OUT = orig_out
