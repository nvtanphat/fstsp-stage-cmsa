from __future__ import annotations

import pytest

from fstsp.config import load_paper_protocol, load_smoke_protocol


def test_paper_protocol_exact_specifications():
    """Verify that paper_protocol.yaml strictly conforms to the published paper."""
    cfg = load_paper_protocol()

    assert cfg.problem.trucks == 1
    assert cfg.problem.drones == 1

    assert cfg.solver.backend == "cplex"
    assert cfg.solver.threads == 8
    assert cfg.solver.mip_emphasis == 5

    assert cfg.exact_time_limits.table1_agatz_maxradius_seconds == 3600.0
    assert cfg.exact_time_limits.table2_agatz_novisit_seconds == 3600.0
    assert cfg.exact_time_limits.table3_cplex_exact_seconds == 7200.0

    assert cfg.cmsa.total_time_limit_seconds == 1800.0
    assert cfg.cmsa.restricted_mip_time_limit_seconds == 15.0
    assert cfg.cmsa.age_limit == 2

    assert cfg.experiments.customer_sizes == [20, 30, 40, 50]
    assert cfg.experiments.instances_per_size == 10
    assert cfg.experiments.table4_age_limits == [2, 5]

    assert cfg.is_paper_protocol() is True


def test_smoke_protocol_distinct_from_paper():
    """Verify that smoke protocol is explicitly identified as not paper reproduction."""
    smoke = load_smoke_protocol()
    assert smoke.cmsa.total_time_limit_seconds < 1800.0
    assert smoke.is_paper_protocol() is False
