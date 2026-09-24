from __future__ import annotations

import pytest

from fstsp.config import load_paper_protocol, load_smoke_protocol
import kaggle.run_experiment as kre


def test_table_specific_time_limits_paper():
    """Verify Table 1-4 time limits adhere strictly to paper protocol specifications."""
    cfg = load_paper_protocol()

    assert cfg.experiments.table1.time_limit_seconds == 3600.0
    assert cfg.experiments.table2.time_limit_seconds == 3600.0
    assert cfg.experiments.table3.exact_time_limit_seconds == 7200.0
    assert cfg.experiments.table3.cmsa_time_limit_seconds == 1800.0
    assert cfg.experiments.table3.restricted_mip_time_limit_seconds == 15.0
    assert cfg.experiments.table4.restricted_mip_time_limit_seconds == 15.0
    assert cfg.experiments.table4.total_time_limit_seconds == 1800.0


def test_runner_settings_reflect_table_time_limits():
    """Verify kaggle runner settings load exact per-table time limits from config."""
    settings = kre._load_settings()

    assert settings["table1_time_limit"] == 3600.0
    assert settings["table2_time_limit"] == 3600.0
    assert settings["table3_exact_time_limit"] == 7200.0
    assert settings["table3_cmsa_time_limit"] == 1800.0
    assert settings["table3_restricted_mip_time_limit"] == 15.0
    assert settings["table4_restricted_mip_time_limit"] == 15.0
    assert settings["table4_total_time_limit"] == 1800.0
