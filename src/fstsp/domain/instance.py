from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import cached_property
import json
import math
import numpy as np


@dataclass(frozen=True)
class FSTSPInstance:
    name: str
    coords: np.ndarray  # customers only, shape (n, 2); depot is implicit at depot_coord
    depot_coord: np.ndarray
    truck_speed: float = 1.0
    drone_speed: float = 2.0
    drone_endurance: float = 120.0
    launch_time: float = 1.0
    recovery_time: float = 1.0
    drone_allowed: np.ndarray | None = None

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        depot = np.asarray(self.depot_coord, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (n, 2)")
        if coords.shape[0] < 1:
            raise ValueError("at least one customer is required")
        if depot.shape != (2,):
            raise ValueError("depot_coord must have shape (2,)")
        if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(depot)):
            raise ValueError("coordinates must be finite")
        numeric = {
            "truck_speed": self.truck_speed,
            "drone_speed": self.drone_speed,
            "drone_endurance": self.drone_endurance,
            "launch_time": self.launch_time,
            "recovery_time": self.recovery_time,
        }
        for field_name, value in numeric.items():
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError(f"{field_name} must be finite")
        if self.truck_speed <= 0 or self.drone_speed <= 0:
            raise ValueError("speeds must be positive")
        if self.drone_endurance <= 0:
            raise ValueError("drone_endurance must be positive")
        if self.launch_time < 0 or self.recovery_time < 0:
            raise ValueError("launch_time and recovery_time must be non-negative")
        allowed = (
            np.ones(coords.shape[0], dtype=bool)
            if self.drone_allowed is None
            else np.asarray(self.drone_allowed, dtype=bool)
        )
        if allowed.shape != (coords.shape[0],):
            raise ValueError("drone_allowed must have shape (n,)")
        object.__setattr__(self, "coords", coords)
        object.__setattr__(self, "depot_coord", depot)
        object.__setattr__(self, "drone_allowed", allowed)

    @property
    def n(self) -> int:
        return int(self.coords.shape[0])

    @property
    def S(self) -> int:
        return 0

    @property
    def E(self) -> int:
        return self.n + 1

    @property
    def customers(self) -> list[int]:
        return list(range(1, self.n + 1))

    @property
    def nodes(self) -> list[int]:
        return list(range(0, self.n + 2))

    @property
    def stages(self) -> list[int]:
        return list(range(0, self.n + 2))

    @cached_property
    def node_coords(self) -> np.ndarray:
        return np.vstack([self.depot_coord, self.coords, self.depot_coord])

    @cached_property
    def truck_time(self) -> np.ndarray:
        xy = self.node_coords
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        return d / self.truck_speed

    @cached_property
    def drone_time(self) -> np.ndarray:
        xy = self.node_coords
        d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        return d / self.drone_speed

    def to_json(self, path: str | Path) -> None:
        payload = {
            "name": self.name,
            "coords": self.coords.tolist(),
            "depot_coord": self.depot_coord.tolist(),
            "truck_speed": self.truck_speed,
            "drone_speed": self.drone_speed,
            "drone_endurance": self.drone_endurance,
            "launch_time": self.launch_time,
            "recovery_time": self.recovery_time,
            "drone_allowed": self.drone_allowed.tolist(),
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "FSTSPInstance":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)
