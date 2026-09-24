from __future__ import annotations

from fstsp.solver.base import SolverBackend
from fstsp.solver.cplex_backend import CplexBackend
from fstsp.solver.highs import HighsBackend


def get_solver_backend(name: str = "highs") -> SolverBackend:
    """Factory function returning the requested SolverBackend instance.

    Supported backends:
    - 'highs' / 'scipy': Open-source HiGHS solver backend.
    - 'cplex' / 'ibm-cplex': IBM ILOG CPLEX commercial solver backend.
    """
    key = name.strip().lower()
    if key in ("highs", "scipy", "scipy-highs"):
        return HighsBackend()
    elif key in ("cplex", "ibm-cplex", "ibm"):
        return CplexBackend()
    else:
        raise ValueError(
            f"Unsupported solver backend: '{name}'. Supported backends are: 'highs', 'cplex'."
        )
