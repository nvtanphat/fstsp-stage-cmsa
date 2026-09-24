from __future__ import annotations

import pytest

from fstsp.config import load_paper_protocol, load_smoke_protocol


def test_paper_table3_budget_consistency():
    """Verify that paper protocol Table 3 adheres to official paper budgets:

    - Exact time limit: 7200 seconds (2 hours)
    - CMSA total time limit: 1800 seconds (30 minutes)
    - Restricted MIP time limit: 15 seconds
    - Age limit: 2
    """
    cfg = load_paper_protocol()
    assert cfg.exact_time_limits.table3_cplex_exact_seconds == 7200.0
    assert cfg.cmsa.total_time_limit_seconds == 1800.0
    assert cfg.cmsa.restricted_mip_time_limit_seconds == 15.0
    assert cfg.cmsa.age_limit == 2


def test_smoke_test_distinct_labeling():
    """Verify that smoke tests are strictly distinguished from paper protocol."""
    smoke_cfg = load_smoke_protocol()
    assert smoke_cfg.experiments.protocol_type == "smoke_test"
    assert smoke_cfg.cmsa.total_time_limit_seconds <= 60.0
    assert smoke_cfg.is_paper_protocol() is False
