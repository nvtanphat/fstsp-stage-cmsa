from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution


def test_cmsa_small_budget_does_not_overrun_by_model_build_seconds():
    # Regression: v0.3 could exceed a 0.35 s budget by >1 s on n=20 because
    # stage-model assembly was outside the solver time limit and non-preemptible.
    inst = generate_uniform_instance(
        20,
        seed=7,
        width=100.0,
        drone_endurance=50.0,
        launch_time=1.0,
        recovery_time=1.0,
        novisit_fraction=0.2,
    )
    budget = 0.35
    sol = solve_cmsa(
        inst,
        total_time=budget,
        mip_time=0.10,
        construct_tsp_time=0.05,
        seed=7,
    )
    assert sol.feasible
    assert validate_solution(inst, sol) == []
    # Small scheduling/OS jitter is acceptable; multi-second overrun is not.
    assert sol.runtime <= budget + 0.20
    assert sol.metadata["budget_overrun_seconds"] <= 0.20
