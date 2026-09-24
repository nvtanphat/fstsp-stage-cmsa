from __future__ import annotations

from pathlib import Path
import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.solution import FSTSPSolution
from fstsp.evaluation.schedule import evaluate_schedule
import kaggle.run_experiment as kre


def test_resume_rejects_solver_seed_mismatch(tmp_path):
    """Verify that cached solutions with a different solver_seed are strictly rejected."""
    inst = generate_uniform_instance(n=4, seed=1)
    tag = "test_seed_tag"
    sol_dir = tmp_path / tag
    sol_dir.mkdir(parents=True)

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
            "total_time": 1800.0,
            "mip_time": 15.0,
            "age_limit": 2,
        },
    )
    sched = evaluate_schedule(inst, sol)
    sol.objective = sched.completion_time
    sol.to_json(sol_dir / "solution.json")

    orig_out = kre.OUT
    kre.OUT = tmp_path
    try:
        # Request seed=2 -> must reject!
        res_mismatch = kre._find_cached_solution(
            tag,
            min_budget=1800.0,
            expected_instance_hash=kre.compute_instance_hash(inst),
            expected_backend="cplex",
            expected_seed=2,
            expected_age_limit=2,
            expected_method="cmsa",
            inst=inst,
        )
        assert res_mismatch is None

        # Request seed=1 -> must accept!
        res_match = kre._find_cached_solution(
            tag,
            min_budget=1800.0,
            expected_instance_hash=kre.compute_instance_hash(inst),
            expected_backend="cplex",
            expected_seed=1,
            expected_age_limit=2,
            expected_method="cmsa",
            inst=inst,
        )
        assert res_match is not None
        assert res_match[0].objective == sched.completion_time
    finally:
        kre.OUT = orig_out
