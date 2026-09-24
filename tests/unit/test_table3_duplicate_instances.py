from __future__ import annotations

import pytest

import kaggle.run_experiment as kre


def test_table3_rejects_duplicate_instance_counting():
    """Verify that duplicate instances are detected and not double-counted towards completion."""
    detail_rows = [
        # Instance (20, 1) duplicated
        {
            "n": 20, "seed": 1, "cmsa_feasible": True, "cmsa_objective": 270.0,
            "cmsa_runtime": 1800.0, "cmsa_total_time": 1800.0, "solver_backend": "cplex",
            "instance_hash": "hash_20_1", "cmsa_validation": "",
        },
        {
            "n": 20, "seed": 1, "cmsa_feasible": True, "cmsa_objective": 269.0,
            "cmsa_runtime": 1800.0, "cmsa_total_time": 1800.0, "solver_backend": "cplex",
            "instance_hash": "hash_20_1", "cmsa_validation": "",
        },
        # Instance (20, 2)
        {
            "n": 20, "seed": 2, "cmsa_feasible": True, "cmsa_objective": 280.0,
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
    # Even though there are 3 rows, only 2 unique instances exist
    assert status["completed"] == 2
    assert any("Duplicate instance" in r for r in status["reasons"])
