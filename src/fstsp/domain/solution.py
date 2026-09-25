from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math

import numpy as np


@dataclass(frozen=True)
class DroneSortie:
    launch_node: int
    customer: int
    recovery_node: int
    launch_stage: int | None = None
    recovery_stage: int | None = None


@dataclass
class FSTSPSolution:
    feasible: bool
    objective: float | None
    truck_route: list[int] = field(default_factory=list)
    drone_sorties: list[DroneSortie] = field(default_factory=list)
    runtime: float = 0.0
    mip_gap: float | None = None
    status: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def _json_safe(value, seen: set | None = None):
        if seen is None:
            seen = set()
        val_id = id(value)
        if isinstance(value, (dict, list, tuple, set)):
            if val_id in seen:
                return {} if isinstance(value, dict) else []
            seen.add(val_id)

        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(k): FSTSPSolution._json_safe(v, seen) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return [FSTSPSolution._json_safe(v, seen) for v in value.tolist()]
        if isinstance(value, (list, tuple, set)):
            return [FSTSPSolution._json_safe(v, seen) for v in value]
        if isinstance(value, Path):
            return value.as_posix()
        if hasattr(value, "item"):
            try:
                return FSTSPSolution._json_safe(value.item(), seen)
            except (ValueError, TypeError):
                pass
        return value

    def to_dict(self) -> dict:
        payload = {
            "feasible": self.feasible,
            "objective": self.objective,
            "truck_route": self.truck_route,
            "drone_sorties": [asdict(s) for s in self.drone_sorties],
            "runtime": self.runtime,
            "mip_gap": self.mip_gap,
            "status": self.status,
            "metadata": self.metadata,
        }
        return self._json_safe(payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict) -> FSTSPSolution:
        sorties = [
            DroneSortie(
                launch_node=int(s["launch_node"]),
                customer=int(s["customer"]),
                recovery_node=int(s["recovery_node"]),
                launch_stage=s.get("launch_stage"),
                recovery_stage=s.get("recovery_stage"),
            )
            for s in data.get("drone_sorties", [])
        ]
        return cls(
            feasible=bool(data.get("feasible", False)),
            objective=data.get("objective"),
            truck_route=[int(x) for x in data.get("truck_route", [])],
            drone_sorties=sorties,
            runtime=float(data.get("runtime", 0.0)),
            mip_gap=data.get("mip_gap"),
            status=str(data.get("status", "unknown")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> FSTSPSolution:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
