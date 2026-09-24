from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import FSTSPSolution


def plot_solution(instance: FSTSPInstance, solution: FSTSPSolution, path: str | Path) -> None:
    xy = instance.node_coords
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(instance.coords[:, 0], instance.coords[:, 1], label="Customers")
    ax.scatter([instance.depot_coord[0]], [instance.depot_coord[1]], marker="s", s=90, label="Depot")
    route = solution.truck_route
    for a, b in zip(route, route[1:]):
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], linewidth=1.5)
    for s in solution.drone_sorties:
        h = s.customer
        ax.plot(
            [xy[s.launch_node, 0], xy[h, 0], xy[s.recovery_node, 0]],
            [xy[s.launch_node, 1], xy[h, 1], xy[s.recovery_node, 1]],
            linestyle="--",
            linewidth=1.2,
        )
    for node in instance.customers:
        ax.annotate(str(node), (xy[node, 0], xy[node, 1]), xytext=(4, 4), textcoords="offset points")
    ax.set_title(f"{instance.name} | objective={solution.objective}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
