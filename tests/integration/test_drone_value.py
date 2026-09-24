import numpy as np
from fstsp.data.generator import generate_uniform_instance
from fstsp.domain.instance import FSTSPInstance
from fstsp.formulation.stage_based import solve_stage_model


def test_allowing_drone_not_worse_than_forcing_truck_only():
    inst = generate_uniform_instance(3, seed=5, width=20, drone_endurance=100)
    truck_only = FSTSPInstance(
        name="truck_only",
        coords=inst.coords,
        depot_coord=inst.depot_coord,
        truck_speed=inst.truck_speed,
        drone_speed=inst.drone_speed,
        drone_endurance=inst.drone_endurance,
        launch_time=inst.launch_time,
        recovery_time=inst.recovery_time,
        drone_allowed=np.zeros(inst.n, dtype=bool),
    )
    a = solve_stage_model(inst, time_limit=10, mip_rel_gap=0.0)
    b = solve_stage_model(truck_only, time_limit=10, mip_rel_gap=0.0)
    assert a.feasible and b.feasible
    assert a.objective <= b.objective + 1e-6
