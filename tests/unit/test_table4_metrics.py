from __future__ import annotations

import pytest

from fstsp.data.generator import generate_uniform_instance
from fstsp.formulation.stage_based import build_stage_model, extract_model_metrics


def test_table4_metrics_decomposition():
    """Verify that variable metrics are properly decomposed and do not confuse fixed-one with fixed-zero."""
    inst = generate_uniform_instance(5, seed=42)
    # Build model with fixed truck route to introduce fixed variables
    # For n=5, nodes are 0..6 (0=inst.S, 1..5=customers, 6=inst.E)
    model = build_stage_model(inst, fixed_truck_route=[inst.S, 1, 2, inst.E])
    metrics = extract_model_metrics(model)

    # Check all required Table 4 keys exist
    assert "n_variables" in metrics
    assert "n_constraints" in metrics
    assert "n_nonzeros" in metrics
    assert "n_fixed_zero_variables" in metrics
    assert "n_fixed_one_variables" in metrics
    assert "n_free_variables" in metrics
    assert "n_active_constraints" in metrics
    assert "n_active_nonzeros" in metrics

    # Partition check: fixed_zero + fixed_one + free == total_variables
    total = metrics["n_variables"]
    fixed_zero = metrics["n_fixed_zero_variables"]
    fixed_one = metrics["n_fixed_one_variables"]
    free = metrics["n_free_variables"]
    assert fixed_zero + fixed_one + free == total

    # Since fixed_truck_route was provided, there must be fixed-one variables
    assert fixed_one > 0
    # And there must be fixed-zero variables
    assert fixed_zero > 0


def test_table4_no_component_fallback():
    """Verify that active_components count is strictly preserved as its own field and NEVER substitutes n_variables."""
    inst = generate_uniform_instance(5, seed=42)
    active_comps = {("x", 0, 1), ("x", 1, 2), ("phi", 3)}
    model = build_stage_model(inst, active_components=active_comps)
    metrics = extract_model_metrics(model, active_components=active_comps)

    assert metrics["n_active_components"] == 3
    # n_variables must reflect the actual mathematical model variables, which is much larger than 3
    assert metrics["n_variables"] > 20
    assert metrics["n_active_components"] != metrics["n_variables"]
