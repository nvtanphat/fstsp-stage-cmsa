from __future__ import annotations

import argparse
from pathlib import Path
import pytest

from fstsp.data.generator import generate_uniform_instance
import kaggle.run_experiment as kre


def test_hash_functions_deterministic():
    """Verify compute_instance_hash and compute_config_hash produce stable hashes."""
    inst1 = generate_uniform_instance(n=10, seed=1)
    inst2 = generate_uniform_instance(n=10, seed=1)
    inst3 = generate_uniform_instance(n=10, seed=2)

    h1 = kre.compute_instance_hash(inst1)
    h2 = kre.compute_instance_hash(inst2)
    h3 = kre.compute_instance_hash(inst3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16

    cfg1 = {"solver_backend": "highs", "threads": 1, "total_time": 1800.0, "mip_time": 15.0, "age_limit": 2}
    cfg2 = {"total_time": 1800.0, "age_limit": 2, "solver_backend": "highs", "threads": 1, "mip_time": 15.0}
    cfg3 = {"solver_backend": "cplex", "threads": 8, "total_time": 1800.0, "mip_time": 15.0, "age_limit": 2}

    assert kre.compute_config_hash(cfg1) == kre.compute_config_hash(cfg2)
    assert kre.compute_config_hash(cfg1) != kre.compute_config_hash(cfg3)


def test_kaggle_settings_protocol_label():
    """Verify Kaggle settings label 1800s budget as paper reproduction and smaller as smoke."""
    args_paper = argparse.Namespace(
        mode=None, solver_backend="highs", threads=None, mip_emphasis=None,
        out=None, sizes=None, seeds=None, total_time=1800.0, mip_time=15.0,
        age_limit=2, exact_time_limit=None, resume=True
    )
    settings_paper = kre._load_settings(args_paper)
    assert settings_paper["protocol_label"] == "paper_1800s"

    args_smoke = argparse.Namespace(
        mode=None, solver_backend="highs", threads=None, mip_emphasis=None,
        out=None, sizes=None, seeds=None, total_time=45.0, mip_time=8.0,
        age_limit=2, exact_time_limit=None, resume=True
    )
    settings_smoke = kre._load_settings(args_smoke)
    assert settings_smoke["protocol_label"] == "smoke_45s"
