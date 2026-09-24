from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.config import load_config
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.metrics import solution_metrics
from fstsp.evaluation.validator import validate_solution


def main() -> None:
    cfg = load_config()

    ap = argparse.ArgumentParser(description="Run FSTSP Benchmark")
    ap.add_argument("--sizes", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--total-time", type=float, default=60)
    ap.add_argument("--mip-time", type=float, default=10)
    ap.add_argument("--age-limit", type=int, default=cfg.cmsa.age_limit)
    ap.add_argument("--backend", "--solver-backend", dest="solver_backend", default=cfg.solver.backend)
    ap.add_argument("--threads", type=int, default=cfg.solver.threads)
    ap.add_argument("--mip-emphasis", type=int, default=cfg.solver.mip_emphasis)
    ap.add_argument("--out", default="artifacts/tables/benchmark.csv")
    args = ap.parse_args()

    rows = []
    invalid = []
    for n in args.sizes:
        for seed in args.seeds:
            inst = generate_uniform_instance(n, seed=seed)
            sol = solve_cmsa(
                inst,
                total_time=args.total_time,
                mip_time=args.mip_time,
                age_limit=args.age_limit,
                seed=seed,
                solver_backend=args.solver_backend,
                threads=args.threads,
                mip_emphasis=args.mip_emphasis,
            )
            issues = validate_solution(inst, sol)
            rows.append(
                {
                    "n": n,
                    "seed": seed,
                    "solver_backend": args.solver_backend,
                    **solution_metrics(sol),
                    "validation": "; ".join(issues),
                }
            )
            if issues:
                invalid.append((n, seed, issues))

    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(frame)
    if invalid:
        raise SystemExit(f"benchmark contains invalid solutions: {invalid}")


if __name__ == "__main__":
    main()
