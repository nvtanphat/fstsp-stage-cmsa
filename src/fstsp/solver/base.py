from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from fstsp.formulation.stage_based import ModelData


@dataclass
class SolverOptions:
    """Standardized options across solver backends."""

    time_limit: float = 60.0
    mip_rel_gap: float = 0.0
    threads: int | None = None
    mip_emphasis: int | None = None  # e.g., 5 for CPLEX feasibility emphasis
    presolve: bool = True
    deadline: float | None = None


@dataclass
class SolverResult:
    """Standardized result returned by any solver backend."""

    status: str
    raw_status: int | str
    success: bool
    x: np.ndarray | None
    objective: float | None
    best_bound: float | None
    mip_gap: float | None
    runtime: float
    node_count: int | None = None
    solver_name: str = "unknown"
    solver_version: str | None = None
    model_metrics: dict = field(default_factory=dict)
    message: str = ""


class SolverBackend(ABC):
    """Abstract base class for FSTSP MILP solver backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the solver backend."""
        pass

    @abstractmethod
    def solve(self, model: ModelData, options: SolverOptions) -> SolverResult:
        """Solve the assembled stage-based MILP model and return a SolverResult."""
        pass

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the underlying solver library and license are available."""
        pass
