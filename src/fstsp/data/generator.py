from __future__ import annotations

import numpy as np
from fstsp.domain.instance import FSTSPInstance


def generate_uniform_instance(
    n: int,
    seed: int = 42,
    width: float = 100.0,
    truck_speed: float = 1.0,
    drone_speed: float = 2.0,
    drone_endurance: float | None = None,
    launch_time: float = 1.0,
    recovery_time: float = 1.0,
    novisit_fraction: float = 0.0,
) -> FSTSPInstance:
    """Generate a deterministic geometric FSTSP instance.

    The paper's 40 new instances are not publicly provided. This generator follows the
    same broad geometric-instance idea, but is an explicit reimplementation choice.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if width <= 0:
        raise ValueError("width must be positive")
    if not 0.0 <= novisit_fraction < 1.0:
        raise ValueError("novisit_fraction must be in [0,1)")
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, width, size=(n, 2))
    depot = np.array([width / 2.0, width / 2.0])
    if drone_endurance is None:
        # 200%-style practical default: long enough for most two-leg sorties.
        max_pair = np.max(
            np.linalg.norm(
                np.vstack([depot, coords])[:, None, :] - np.vstack([depot, coords])[None, :, :],
                axis=2,
            )
        )
        drone_endurance = 2.0 * max_pair / drone_speed + recovery_time
    allowed = np.ones(n, dtype=bool)
    k = int(round(n * novisit_fraction))
    if k:
        blocked = rng.choice(n, size=k, replace=False)
        allowed[blocked] = False
    return FSTSPInstance(
        name=f"uniform_n{n}_seed{seed}",
        coords=coords,
        depot_coord=depot,
        truck_speed=truck_speed,
        drone_speed=drone_speed,
        drone_endurance=float(drone_endurance),
        launch_time=launch_time,
        recovery_time=recovery_time,
        drone_allowed=allowed,
    )
