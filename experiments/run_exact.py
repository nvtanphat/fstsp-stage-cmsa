from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model
from fstsp.visualization.route import plot_solution


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=float, default=30)
    ap.add_argument(
        "--mip-rel-gap",
        type=float,
        default=0.0,
        help="Relative MIP gap. Keep 0.0 for an exact/proven-optimal run when time permits.",
    )
    ap.add_argument("--out", default="artifacts/runs/exact")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    inst = generate_uniform_instance(args.n, seed=args.seed)
    sol = solve_stage_model(
        inst, time_limit=args.time_limit, mip_rel_gap=args.mip_rel_gap
    )
    issues = validate_solution(inst, sol)
    inst.to_json(out / "instance.json")
    sol.to_json(out / "solution.json")
    if sol.feasible:
        plot_solution(inst, sol, out / "route.png")
    print(sol.to_dict())
    print("validation:", issues or "OK")
    if issues:
        raise SystemExit("result failed independent schedule validation")

if __name__ == "__main__":
    main()
