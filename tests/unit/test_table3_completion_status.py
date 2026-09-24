from __future__ import annotations

import pytest

import kaggle.run_experiment as kre


def test_table3_completion_status_empty():
    """Verify empty run set results in NOT_STARTED."""
    status = kre.evaluate_table3_completion_status([])
    assert status["status"] == "NOT_STARTED"
    assert status["completed"] == 0
    assert status["missing"] == 40
    assert "Completed: 0/40" in status["summary_text"]
    assert "Protocol status: NOT_STARTED" in status["summary_text"]


def test_table3_completion_status_partial_12():
    """Verify 12 completed instances produce PARTIAL with exact summary text matching audit spec."""
    detail_rows = []
    # 12 instances: n=20 (10 seeds), n=30 (2 seeds)
    for s in range(1, 11):
        detail_rows.append({
            "n": 20, "seed": s, "cmsa_feasible": True, "cmsa_objective": 270.0,
            "cmsa_runtime": 1800.0, "cmsa_total_time": 1800.0, "solver_backend": "cplex",
            "instance_hash": f"hash_20_{s}", "cmsa_validation": "",
        })
    for s in range(1, 3):
        detail_rows.append({
            "n": 30, "seed": s, "cmsa_feasible": True, "cmsa_objective": 350.0,
            "cmsa_runtime": 1800.0, "cmsa_total_time": 1800.0, "solver_backend": "cplex",
            "instance_hash": f"hash_30_{s}", "cmsa_validation": "",
        })

    status = kre.evaluate_table3_completion_status(
        detail_rows,
        expected_sizes=[20, 30, 40, 50],
        expected_seeds_per_size=10,
        required_backend="cplex",
        required_budget=1800.0,
    )
    assert status["status"] == "PARTIAL"
    assert status["completed"] == 12
    assert status["missing"] == 28
    assert "Completed: 12/40" in status["summary_text"]
    assert "Missing: 28" in status["summary_text"]
    assert "Protocol status: PARTIAL" in status["summary_text"]


def test_table3_completion_status_full_40():
    """Verify full 40 valid instances produce COMPLETE."""
    detail_rows = []
    for n in [20, 30, 40, 50]:
        for s in range(1, 11):
            detail_rows.append({
                "n": n, "seed": s, "cmsa_feasible": True, "cmsa_objective": 300.0,
                "cmsa_runtime": 1800.0, "cmsa_total_time": 1800.0, "solver_backend": "cplex",
                "instance_hash": f"hash_{n}_{s}", "cmsa_validation": "",
            })

    status = kre.evaluate_table3_completion_status(
        detail_rows,
        expected_sizes=[20, 30, 40, 50],
        expected_seeds_per_size=10,
        required_backend="cplex",
        required_budget=1800.0,
    )
    assert status["status"] == "COMPLETE"
    assert status["completed"] == 40
    assert status["missing"] == 0
    assert status["is_complete"] is True
    assert "Completed: 40/40" in status["summary_text"]
    assert "Protocol status: COMPLETE" in status["summary_text"]
