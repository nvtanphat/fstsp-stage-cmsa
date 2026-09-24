from __future__ import annotations

from pathlib import Path
import re
import numpy as np

from fstsp.domain.instance import FSTSPInstance

_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def load_geometric_instance(path: str | Path, name: str | None = None) -> FSTSPInstance:
    """Parse pcbouman-eur/TSP-D-Instances geometric benchmark files.

    The public format stores *cost/time factors per unit Euclidean distance* for
    truck and drone, followed by the number of nodes (including the depot), then
    x/y/id triplets. Optional #MAXFLY is an absolute drone-distance limit and
    #NOVISIT marks node indices forbidden to the drone.

    Internally FSTSPInstance exposes speeds, so factor f is represented as speed
    1/f such that distance / speed == distance * f.
    """
    p = Path(path)
    raw = _COMMENT.sub("", p.read_text(encoding="utf-8", errors="ignore"))
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    maxfly_distance: float | None = None
    novisit: set[int] = set()
    content: list[str] = []
    for line in lines:
        if line.startswith("#MAXFLY"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed #MAXFLY line in {p}: {line}")
            val = parts[1].strip()
            if val.lower() in ("infinity", "inf", "+inf"):
                maxfly_distance = None
            else:
                maxfly_distance = float(val)
        elif line.startswith("#NOVISIT"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed #NOVISIT line in {p}: {line}")
            novisit.add(int(parts[1]))
        elif not line.startswith("#"):
            content.extend(line.split())

    if len(content) < 3:
        raise ValueError(f"Cannot parse {p}: missing header")

    truck_factor = float(content[0])
    drone_factor = float(content[1])
    if truck_factor <= 0 or drone_factor <= 0:
        raise ValueError(f"Truck/drone factors must be positive in {p}")

    n_nodes = int(float(content[2]))
    if n_nodes < 2:
        raise ValueError(f"Expected depot + at least one customer in {p}")

    triples = content[3 : 3 + 3 * n_nodes]
    if len(triples) != 3 * n_nodes:
        raise ValueError(f"Expected {n_nodes} node definitions in {p}")

    xy: list[tuple[float, float]] = []
    for idx in range(n_nodes):
        x = float(triples[3 * idx])
        y = float(triples[3 * idx + 1])
        # triples[3*idx+2] is the textual identifier; ordering is authoritative
        # for #NOVISIT according to the public benchmark format.
        xy.append((x, y))

    xy_arr = np.asarray(xy, dtype=float)
    depot = xy_arr[0]
    customers = xy_arr[1:]

    truck_speed = 1.0 / truck_factor
    drone_speed = 1.0 / drone_factor

    allowed = np.ones(len(customers), dtype=bool)
    for node_idx in novisit:
        # Public files use node index 0 for the depot and 1.. for locations.
        if 1 <= node_idx < n_nodes:
            allowed[node_idx - 1] = False
        elif node_idx != 0:
            raise ValueError(f"#NOVISIT index {node_idx} outside 0..{n_nodes - 1} in {p}")

    if maxfly_distance is None:
        # The benchmark README states that 200% of the maximum pairwise distance
        # is sufficient to represent an unrestricted single drone operation (two
        # legs). Use that finite equivalent rather than 1e9 to avoid catastrophic
        # Big-M scaling/numerical conditioning in the stage MILP.
        pairwise = np.linalg.norm(xy_arr[:, None, :] - xy_arr[None, :, :], axis=2)
        max_pair_distance = float(np.max(pairwise))
        maxfly_distance = 2.0 * max_pair_distance

    # Public benchmark launch/recovery handling times are not encoded in the files.
    # We therefore use zero, matching the original geometric benchmark semantics.
    endurance_time = float(maxfly_distance * drone_factor)

    return FSTSPInstance(
        name=name or p.stem,
        coords=customers,
        depot_coord=depot,
        truck_speed=truck_speed,
        drone_speed=drone_speed,
        drone_endurance=endurance_time,
        launch_time=0.0,
        recovery_time=0.0,
        drone_allowed=allowed,
    )
