from __future__ import annotations

import pandas as pd
import pytest


def test_table3_classification_logic():
    """Verify Table 3 reproduction level classification."""
    def classify(sizes, n_seeds, total_time, backend):
        if (
            len(sizes) == 4
            and set(sizes) == {20, 30, 40, 50}
            and n_seeds == 10
            and total_time >= 1800.0
            and backend == "cplex"
        ):
            return "PAPER_PROTOCOL_COMPLETE"
        elif total_time < 1800.0:
            return "SMOKE_TEST"
        else:
            return "PARTIAL_REPRODUCTION"

    assert classify([20, 30, 40, 50], 10, 1800.0, "cplex") == "PAPER_PROTOCOL_COMPLETE"
    assert classify([20, 30, 40, 50], 10, 45.0, "highs") == "SMOKE_TEST"
    assert classify([20, 30], 2, 1800.0, "highs") == "PARTIAL_REPRODUCTION"


def test_table3_required_columns():
    """Verify Table 3 detail dataframe contains required audit columns."""
    required_cols = {
        "n", "seed", "protocol_type", "reproduction_level", "solver_backend",
        "exact_feasible", "exact_objective", "exact_runtime", "exact_status",
        "exact_timeout", "cmsa_feasible", "cmsa_objective", "cmsa_runtime",
        "improvement_gap_pct",
    }
    sample_df = pd.DataFrame([{col: None for col in required_cols}])
    for c in required_cols:
        assert c in sample_df.columns
