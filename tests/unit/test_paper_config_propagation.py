from __future__ import annotations

import pytest
from pathlib import Path

from fstsp.config import (
    AppConfig,
    load_config,
    load_paper_protocol,
    load_smoke_protocol,
)


def test_load_paper_protocol():
    config = load_paper_protocol()
    assert isinstance(config, AppConfig)
    assert config.solver.backend == "cplex"
    assert config.solver.threads == 8
    assert config.solver.mip_emphasis == 5

    # Exact time limits
    assert config.exact_time_limits.table1_agatz_maxradius_seconds == 3600.0
    assert config.exact_time_limits.table2_agatz_novisit_seconds == 3600.0
    assert config.exact_time_limits.table3_cplex_exact_seconds == 7200.0

    # CMSA parameters
    assert config.cmsa.age_limit == 2
    assert config.cmsa.restricted_mip_time_limit_seconds == 15.0
    assert config.cmsa.total_time_limit_seconds == 1800.0
    assert config.cmsa.truck_sample_ratio == 0.65

    # Protocol validation
    assert config.is_paper_protocol() is True


def test_load_smoke_protocol():
    config = load_smoke_protocol()
    assert isinstance(config, AppConfig)
    assert config.solver.backend == "highs"
    assert config.cmsa.total_time_limit_seconds == 10.0
    assert config.experiments.protocol_type == "smoke_test"
    assert config.is_paper_protocol() is False


def test_config_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_path_12345.yaml")
