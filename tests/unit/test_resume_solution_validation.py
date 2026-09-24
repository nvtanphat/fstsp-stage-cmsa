from __future__ import annotations

from pathlib import Path
import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.solution import FSTSPSolution
from fstsp.evaluation.schedule import evaluate_schedule
import kaggle.run_experiment as kre


def test_resume_rejects_missing_file(tmp_path):
    """Verify that cached entry with missing solution.json is rejected."""
    inst = generate_uniform_instance(n=6, seed=1)
    tag = "test_missing_file_tag"
    (tmp_path / tag).mkdir(parents=True)

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        res = kre._find_cached_solution(
            tag,
            expected_instance_hash=kre.compute_instance_hash(inst),
            inst=inst,
        )
        assert res is None
    finally:
        kre.OUT = orig_out


def test_resume_rejects_infeasible_solution(tmp_path):
    """Verify that cached solution with feasible=False is rejected."""
    inst = generate_uniform_instance(n=6, seed=1)
    tag = "test_infeasible_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    sol = FSTSPSolution(
        feasible=False,
        objective=None,
        runtime=10.0,
        truck_route=[],
        status="infeasible",
        metadata={
            "schema_version": kre.SCHEMA_VERSION,
            "instance_hash": kre.compute_instance_hash(inst),
            "solver_backend": "cplex",
            "solver_seed": 1,
            "method": "cmsa",
        },
    )
    sol.to_json(sol_dir / "solution.json")

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        res = kre._find_cached_solution(
            tag,
            expected_instance_hash=kre.compute_instance_hash(inst),
            inst=inst,
        )
        assert res is None
    finally:
        kre.OUT = orig_out


def test_resume_rejects_invalid_solution_structure(tmp_path):
    """Verify that cached solution failing independent validation is rejected."""
    inst = generate_uniform_instance(n=6, seed=1)
    tag = "test_invalid_structure_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    # Missing customers 2, 3, 4, 5
    sol = FSTSPSolution(
        feasible=True,
        objective=50.0,
        runtime=10.0,
        truck_route=[0, 1, 7],
        drone_sorties=[],
        status="cmsa_completed",
        metadata={
            "schema_version": kre.SCHEMA_VERSION,
            "instance_hash": kre.compute_instance_hash(inst),
            "solver_backend": "cplex",
            "solver_seed": 1,
            "method": "cmsa",
            "age_limit": 2,
            "total_time": 1800.0,
            "mip_time": 15.0,
        },
    )
    sol.to_json(sol_dir / "solution.json")

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        res = kre._find_cached_solution(
            tag,
            expected_instance_hash=kre.compute_instance_hash(inst),
            expected_backend="cplex",
            expected_seed=1,
            expected_age_limit=2,
            expected_method="cmsa",
            inst=inst,
        )
        assert res is None
    finally:
        kre.OUT = orig_out


def test_resume_accepts_fully_valid_solution(tmp_path):
    """Verify that cached solution satisfying all criteria and passing validation is accepted."""
    inst = generate_uniform_instance(n=4, seed=1)
    tag = "test_valid_solution_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

    # Valid all-truck solution covering 0, 1, 2, 3, 4, 5
    sol = FSTSPSolution(
        feasible=True,
        objective=0.0,
        runtime=1800.0,
        truck_route=[0, 1, 2, 3, 4, 5],
        drone_sorties=[],
        status="cmsa_completed",
        metadata={
            "schema_version": kre.SCHEMA_VERSION,
            "instance_hash": kre.compute_instance_hash(inst),
            "solver_backend": "cplex",
            "solver_seed": 1,
            "method": "cmsa",
            "age_limit": 2,
            "total_time": 1800.0,
            "mip_time": 15.0,
        },
    )
    sched = evaluate_schedule(inst, sol)
    sol.objective = sched.completion_time
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
        assert res is not None
        assert res[0].objective == sched.completion_time
    finally:
        kre.OUT = orig_out
