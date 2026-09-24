from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution
from fstsp.visualization.route import plot_solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--total-time", type=float, default=60)
    ap.add_argument("--mip-time", type=float, default=10)
    ap.add_argument("--age-limit", type=int, default=2)
    ap.add_argument("--out", default="artifacts/runs/cmsa")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    inst = generate_uniform_instance(args.n, seed=args.seed)
    sol = solve_cmsa(inst, total_time=args.total_time, mip_time=args.mip_time, age_limit=args.age_limit, seed=args.seed)
    issues = validate_solution(inst, sol)
    inst.to_json(out / "instance.json")
    sol.to_json(out / "solution.json")
    if sol.feasible:
        plot_solution(inst, sol, out / "route.png")
    print({"feasible": sol.feasible, "objective": sol.objective, "runtime": sol.runtime, "status": sol.status, "truck_route": sol.truck_route, "drone_sorties": [vars(x) for x in sol.drone_sorties], "iterations": sol.metadata.get("iterations")})
    print("validation:", issues or "OK")
    if issues:
        raise SystemExit("result failed independent schedule validation")

if __name__ == "__main__":
    main()
