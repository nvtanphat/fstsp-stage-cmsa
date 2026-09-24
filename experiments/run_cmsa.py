from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.config import load_config, load_paper_protocol, load_smoke_protocol
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.validator import validate_solution
from fstsp.visualization.route import plot_solution


def main() -> None:
    ap = argparse.ArgumentParser(description="Run CMSA solver for FSTSP")
    ap.add_argument("--n", type=int, default=12, help="Number of customers")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for instance generation")
    ap.add_argument("--total-time", type=float, default=None, help="CMSA total wall-clock time limit in seconds")
    ap.add_argument("--mip-time", type=float, default=None, help="Restricted MIP subproblem time limit in seconds")
    ap.add_argument("--age-limit", type=int, default=None, help="Component age limit before destruction (Algorithm 1)")
    ap.add_argument(
        "--protocol",
        choices=["paper", "smoke"],
        default="paper",
        help="Protocol preset ('paper' for 1800s total / 15s MIP / age=2, 'smoke' for fast CI)",
    )
    ap.add_argument("--config", type=str, default=None, help="Path to custom config YAML file")
    ap.add_argument("--solver", type=str, default=None, help="Solver backend ('cplex' or 'highs')")
    ap.add_argument("--threads", type=int, default=None, help="Solver thread count")
    ap.add_argument("--mip-emphasis", type=int, default=None, help="MIP emphasis (5 for CPLEX feasibility)")
    ap.add_argument("--out", default="artifacts/runs/cmsa", help="Output directory")
    args = ap.parse_args()

    # Load configuration
    if args.config:
        cfg = load_config(args.config)
    elif args.protocol == "smoke":
        cfg = load_smoke_protocol()
    else:
        cfg = load_paper_protocol()

    total_time = args.total_time if args.total_time is not None else cfg.cmsa.total_time_limit_seconds
    mip_time = args.mip_time if args.mip_time is not None else cfg.cmsa.restricted_mip_time_limit_seconds
    age_limit = args.age_limit if args.age_limit is not None else cfg.cmsa.age_limit
    solver_backend = args.solver or cfg.solver.backend
    threads = args.threads or cfg.solver.threads
    mip_emphasis = args.mip_emphasis if args.mip_emphasis is not None else cfg.solver.mip_emphasis

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inst = generate_uniform_instance(args.n, seed=args.seed)
    sol = solve_cmsa(
        inst,
        total_time=total_time,
        mip_time=mip_time,
        age_limit=age_limit,
        truck_sample_ratio=cfg.cmsa.truck_sample_ratio,
        seed=args.seed,
        solver_backend=solver_backend,
        threads=threads,
        mip_emphasis=mip_emphasis,
    )
    issues = validate_solution(inst, sol)
    inst.to_json(out / "instance.json")
    sol.to_json(out / "solution.json")
    if sol.feasible:
        plot_solution(inst, sol, out / "route.png")

    print({
        "feasible": sol.feasible,
        "objective": sol.objective,
        "runtime": sol.runtime,
        "status": sol.status,
        "solver_backend": solver_backend,
        "truck_route": sol.truck_route,
        "drone_sorties": [vars(x) for x in sol.drone_sorties],
        "iterations": sol.metadata.get("iterations"),
    })
    print("validation:", issues or "OK")
    if issues:
        raise SystemExit("result failed independent schedule validation")


if __name__ == "__main__":
    main()
