from fstsp.data.generator import generate_uniform_instance
from fstsp.formulation.stage_based import solve_stage_model


def test_exact_default_requests_zero_gap_and_records_proof_flag():
    inst = generate_uniform_instance(2, seed=11, launch_time=0.0, recovery_time=0.0)
    sol = solve_stage_model(inst, time_limit=10.0)
    assert sol.feasible
    assert sol.metadata["requested_mip_rel_gap"] == 0.0
    assert isinstance(sol.metadata["proven_optimal"], bool)
    if sol.metadata["proven_optimal"]:
        assert sol.mip_gap is not None and sol.mip_gap <= 1e-9
