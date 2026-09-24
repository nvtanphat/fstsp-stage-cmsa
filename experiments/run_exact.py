from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fstsp.config import load_config, load_paper_protocol, load_smoke_protocol
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model
from fstsp.visualization.route import plot_solution


def main() -> None:
    ap = argparse.ArgumentParser(description="Run exact 2-index stage-based MILP solver")
    ap.add_argument("--n", type=int, default=6, help="Number of customers")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for instance generation")
    ap.add_argument("--time-limit", type=float, default=None, help="Solve time limit in seconds")
    ap.add_argument(
        "--mip-rel-gap",
        type=float,
        default=0.0,
        help="Relative MIP gap. Keep 0.0 for an exact/proven-optimal run.",
    )
    ap.add_argument(
        "--protocol",
        choices=["paper", "smoke"],
        default="paper",
        help="Protocol preset ('paper' for 7200s Table 3 / 3600s Table 1-2, 'smoke' for fast CI)",
    )
    ap.add_argument("--config", type=str, default=None, help="Path to custom config YAML file")
    ap.add_argument("--solver", type=str, default=None, help="Solver backend ('cplex' or 'highs')")
    ap.add_argument("--threads", type=int, default=None, help="Solver thread count")
    ap.add_argument("--mip-emphasis", type=int, default=None, help="MIP emphasis (5 for CPLEX feasibility)")
    ap.add_argument("--out", default="artifacts/runs/exact", help="Output directory")
    args = ap.parse_args()

    # Load configuration
    if args.config:
        cfg = load_config(args.config)
    elif args.protocol == "smoke":
        cfg = load_smoke_protocol()
    else:
        cfg = load_paper_protocol()

    solver_backend = args.solver or cfg.solver.backend
    threads = args.threads or cfg.solver.threads
    mip_emphasis = args.mip_emphasis if args.mip_emphasis is not None else cfg.solver.mip_emphasis
    time_limit = args.time_limit if args.time_limit is not None else cfg.exact_time_limits.table3_cplex_exact_seconds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inst = generate_uniform_instance(args.n, seed=args.seed)
    sol = solve_stage_model(
        inst,
        time_limit=time_limit,
        mip_rel_gap=args.mip_rel_gap,
        solver_backend=solver_backend,
        threads=threads,
        mip_emphasis=mip_emphasis,
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
