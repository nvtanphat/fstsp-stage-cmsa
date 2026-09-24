from __future__ import annotations

import pytest

import kaggle.run_experiment as kre


def test_table3_isolates_smoke_test_results():
    """Verify that 45s smoke test runs cannot be conflated into 1800s paper protocol completion."""
    detail_rows = [
        # Instance run under 45s smoke test
        {
            "n": 20, "seed": 1, "cmsa_feasible": True, "cmsa_objective": 280.0,
            "cmsa_runtime": 45.0, "cmsa_total_time": 45.0, "solver_backend": "cplex",
            "instance_hash": "hash_20_1", "cmsa_validation": "",
        },
        # Instance run under 1800s paper protocol
        {
            "n": 20, "seed": 2, "cmsa_feasible": True, "cmsa_objective": 275.0,
            "cmsa_runtime": 1800.0, "cmsa_total_time": 1800.0, "solver_backend": "cplex",
            "instance_hash": "hash_20_2", "cmsa_validation": "",
        },
    ]

    status = kre.evaluate_table3_completion_status(
        detail_rows,
        expected_sizes=[20],
        expected_seeds_per_size=2,
        required_backend="cplex",
        required_budget=1800.0,
    )
    # The 45s run must be rejected and not counted towards 1800s completion
    assert status["completed"] == 1
    assert status["missing"] == 1
    assert status["status"] == "PARTIAL"
    assert any("smoke test isolated" in r for r in status["reasons"])
