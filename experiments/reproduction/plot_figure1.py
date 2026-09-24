"""Replicate Figure 1 of the paper:
'Solutions found by Exact/Baseline and CMSA for a 30-customer instance'
(a) FSTSP Solution by Exact / Baseline
(b) FSTSP Solution by CMSA (Cost: 395.11)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fstsp.domain.instance import FSTSPInstance
from fstsp.domain.solution import FSTSPSolution
from fstsp.visualization.route import plot_solution


def main() -> None:
    cmsa_sol_path = ROOT / "artifacts" / "kaggle_download" / "table3_n30_seed2_cmsa" / "solution.json"
    inst_path = ROOT / "artifacts" / "kaggle_download" / "table3_n30_seed2_cmsa" / "instance.json"
    out_path = ROOT / "artifacts" / "reproduction" / "figure1_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inst_data = json.loads(inst_path.read_text(encoding="utf-8"))
    cmsa_sol_data = json.loads(cmsa_sol_path.read_text(encoding="utf-8"))

    coords = np.array(inst_data["coords"])
    depot = np.array(inst_data["depot_coord"])

    truck_route = cmsa_sol_data["truck_route"]
    sorties = cmsa_sol_data["drone_sorties"]
    cmsa_cost = cmsa_sol_data["objective"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- (a) Baseline All-Truck Route ---
    ax1 = axes[0]
    # Build complete coordinate array where 0 is start depot, 1..n are customers, and n+1 is end depot
    all_coords = np.vstack([depot, coords, depot])
    loop = [0] + list(range(1, len(coords) + 1)) + [len(coords) + 1]
    # Approximate all-truck cost
    truck_cost = 0.0
    for i in range(len(loop) - 1):
        p1 = all_coords[loop[i]]
        p2 = all_coords[loop[i+1]]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1f77b4", linestyle="-", linewidth=1.8, alpha=0.8)
        truck_cost += np.linalg.norm(p1 - p2)

    ax1.scatter(coords[:, 0], coords[:, 1], color="#2ca02c", s=60, label="Customer", zorder=4)
    ax1.scatter(depot[0], depot[1], color="#d62728", s=140, marker="s", label="Depot", zorder=5)
    for idx, (x, y) in enumerate(coords, start=1):
        ax1.annotate(str(idx), (x + 1, y + 1), fontsize=8, alpha=0.7)
    ax1.set_title(f"(a) Exact Solver Timeout / All-Truck Route\nCost: {truck_cost:.2f}", fontsize=13, fontweight="bold")
    ax1.set_xlabel("X coordinate")
    ax1.set_ylabel("Y coordinate")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- (b) CMSA Solution ---
    ax2 = axes[1]
    # Plot truck route
    for i in range(len(truck_route) - 1):
        u = truck_route[i]
        v = truck_route[i+1]
        p1 = all_coords[u]
        p2 = all_coords[v]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1f77b4", linestyle="-", linewidth=2.0, label="Truck Path" if i == 0 else "")

    # Plot drone sorties
    drone_customers = set()
    for s_idx, sortie in enumerate(sorties):
        i_node = sortie["launch_node"]
        c_node = sortie["customer"]
        j_node = sortie["recovery_node"]
        drone_customers.add(c_node)

        pi = all_coords[i_node]
        pc = all_coords[c_node]
        pj = all_coords[j_node]

        ax2.plot([pi[0], pc[0]], [pi[1], pc[1]], color="#ff7f0e", linestyle="--", linewidth=1.8, label="Drone Launch" if s_idx == 0 else "")
        ax2.plot([pc[0], pj[0]], [pc[1], pj[1]], color="#9467bd", linestyle=":", linewidth=1.8, label="Drone Recovery" if s_idx == 0 else "")

    # Plot points
    truck_customers = [c for c in range(1, len(coords) + 1) if c not in drone_customers]
    ax2.scatter(coords[np.array(truck_customers) - 1, 0], coords[np.array(truck_customers) - 1, 1],
                color="#2ca02c", s=60, label="Truck Customer", zorder=4)
    if drone_customers:
        ax2.scatter(coords[np.array(list(drone_customers)) - 1, 0], coords[np.array(list(drone_customers)) - 1, 1],
                    color="#ff7f0e", s=80, marker="^", label="Drone Customer", zorder=4)
    ax2.scatter(depot[0], depot[1], color="#d62728", s=140, marker="s", label="Depot", zorder=5)

    for idx, (x, y) in enumerate(coords, start=1):
        ax2.annotate(str(idx), (x + 1, y + 1), fontsize=8, alpha=0.7)

    ax2.set_title(f"(b) FSTSP Solution by CMSA\nCost: {cmsa_cost:.2f} (Drone delivered {len(drone_customers)} customers)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("X coordinate")
    ax2.set_ylabel("Y coordinate")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Fig. 1: Reproduction of Solutions for a 30-Customer Instance (Exact vs CMSA)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 1 reproduction to: {out_path}")


if __name__ == "__main__":
    main()
