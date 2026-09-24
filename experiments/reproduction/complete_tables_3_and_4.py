"""Complete reproduction of Paper Tables 3 and 4 by finishing remaining instances.

Loads existing progress from Kaggle Version 6 (n=20 all 10 seeds, n=30 all 10 seeds,
n=40 all 10 seeds, n=50 seeds 1..5) and finishes n=50 seeds 6..10 + Table 4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd

from fstsp.config import load_config, load_paper_protocol, load_smoke_protocol
from fstsp.data.generator import generate_uniform_instance
from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.evaluation.validator import validate_solution
from fstsp.visualization.route import plot_solution

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "artifacts" / "kaggle_v6_download"
OUT_DIR = ROOT / "artifacts" / "reproduction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

paper_baselines = {
    20: {"cplex": 300.83, "csma": 277.18, "gap": -7.86},
    30: {"cplex": 619.49, "csma": 353.42, "gap": -42.95},
    40: {"cplex": "-", "csma": 422.87, "gap": None},
    50: {"cplex": "-", "csma": 503.86, "gap": None},
}

paper_table4_ref = {
    20: {"age2": (808, 178, 3312), "age5": (8814, 1378, 46627)},
    30: {"age2": (3263, 796, 19304), "age5": (18216, 2954, 122796)},
    40: {"age2": (12686, 2928, 110669), "age5": (30012, 5036, 246644)},
    50: {"age2": (14638, 3425, 153145), "age5": (48327, 7971, 467232)},
}


def complete_table3(
    total_time: float = 45.0,
    mip_time: float = 8.0,
    age_limit: int = 2,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
) -> pd.DataFrame:
    print(f"=== Completing Table 3 (Finishing n=50 seeds 6..10) [backend={solver_backend}] ===")
    t3_path = IN_DIR / "table3_details.csv"
    if t3_path.exists():
        df_details = pd.read_csv(t3_path)
        detail_rows = df_details.to_dict(orient="records")
    else:
        detail_rows = []

    # Check which (n, seed) are present
    completed = set()
    for r in detail_rows:
        if pd.notna(r.get("cmsa_objective")):
            completed.add((int(r["n"]), int(r["seed"])))

    print(f"Loaded {len(completed)} completed Table 3 instances.")

    for seed in range(6, 11):
        if (50, seed) in completed:
            print(f"n=50 seed={seed} already completed.")
            continue
        print(f"Solving n=50 seed={seed} with CMSA (total_time={total_time}s, mip_time={mip_time}s, age_limit={age_limit})...")
        inst = generate_uniform_instance(n=50, seed=seed)
        t0 = time.perf_counter()
        sol = solve_cmsa(
            inst,
            total_time=total_time,
            mip_time=mip_time,
            age_limit=age_limit,
            seed=seed,
            solver_backend=solver_backend,
            threads=threads,
            mip_emphasis=mip_emphasis,
        )
        rt = time.perf_counter() - t0
        issues = validate_solution(inst, sol)
        print(f"  -> seed={seed}: obj={sol.objective:.2f} ({rt:.1f}s), feasible={sol.feasible}, issues={issues}")

        run_dir = OUT_DIR / f"table3_n50_seed{seed}_cmsa"
        run_dir.mkdir(parents=True, exist_ok=True)
        inst.to_json(run_dir / "instance.json")
        sol.to_json(run_dir / "solution.json")
        if sol.feasible and not issues:
            plot_solution(inst, sol, run_dir / "route.png")

        detail_rows.append({
            "n": 50,
            "seed": seed,
            "exact_feasible": False,
            "exact_objective": None,
            "exact_runtime": 0.0,
            "exact_status": "NOT_RUN",
            "exact_timeout": False,
            "cmsa_feasible": sol.feasible,
            "cmsa_objective": sol.objective,
            "cmsa_runtime": rt,
            "improvement_gap_pct": None,
            "solver_backend": solver_backend,
        })

    df_full_details = pd.DataFrame(detail_rows)
    df_full_details.to_csv(OUT_DIR / "table3_details.csv", index=False)
    print(f"Saved full {len(df_full_details)} rows to {OUT_DIR / 'table3_details.csv'}")

    # Build summary table
    summary_rows = []
    for n in [20, 30, 40, 50]:
        sub = df_full_details[df_full_details["n"] == n]
        cmsa_objs = [x for x in sub["cmsa_objective"] if pd.notna(x)]
        avg_cmsa = f"{np.mean(cmsa_objs):.2f}" if cmsa_objs else "-"
        pb = paper_baselines.get(n, {})
        summary_rows.append({
            "n": n,
            "Avg. Exact Obj (Our)": "-",
            "Avg. CMSA Obj (Our)": avg_cmsa,
            "Avg. Improvement (Our)": "-",
            "Paper CPLEX Obj": pb.get("cplex", "-"),
            "Paper CSMA Obj": pb.get("csma", "-"),
            "Paper Improvement": f"{pb.get('gap')}%" if pb.get("gap") else "-",
        })
    df_t3_summary = pd.DataFrame(summary_rows)
    df_t3_summary.to_csv(OUT_DIR / "table3_reproduction.csv", index=False)
    print("\n--- Completed Table 3 Reproduction Summary ---")
    print(df_t3_summary.to_string(index=False))
    return df_t3_summary


def run_table4(
    total_time: float = 15.0,
    mip_time: float = 5.0,
    solver_backend: str = "highs",
    threads: int | None = None,
    mip_emphasis: int | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    print(f"\n=== Executing Table 4 (age=2 vs age=5 model sizing) [backend={solver_backend}] ===")
    detail_rows = []
    iteration_rows = []
    summary_rows = []
    sizes = [20, 30, 40, 50]
    if seeds is None:
        seeds = [1, 2]

    for n in sizes:
        print(f"\nProfiling model size for n={n}...")
        stats_by_age = {2: {"cons": [], "vars": [], "coefs": []}, 5: {"cons": [], "vars": [], "coefs": []}}

        for age in [2, 5]:
            for seed in seeds:
                inst = generate_uniform_instance(n=n, seed=seed)
                t0 = time.perf_counter()
                sol = solve_cmsa(
                    inst,
                    total_time=total_time,
                    mip_time=mip_time,
                    age_limit=age,
                    seed=seed,
                    solver_backend=solver_backend,
                    threads=threads,
                    mip_emphasis=mip_emphasis,
                )
                rt = time.perf_counter() - t0

                meta = sol.metadata or {}
                hist = meta.get("history", [])

                # Record per-iteration metrics
                for it_idx, it_data in enumerate(hist):
                    iteration_rows.append({
                        "n": n,
                        "age": age,
                        "seed": seed,
                        "iteration": it_idx + 1,
                        "original_variables": it_data.get("original_variables", it_data.get("n_variables")),
                        "original_constraints": it_data.get("original_constraints", it_data.get("n_constraints")),
                        "original_nonzeros": it_data.get("original_nonzeros", it_data.get("n_nonzeros")),
                        "fixed_zero_variables": it_data.get("fixed_zero_variables", 0),
                        "fixed_one_variables": it_data.get("fixed_one_variables", 0),
                        "free_variables": it_data.get("free_variables", it_data.get("n_active_variables", 0)),
                        "active_constraints_before_presolve": it_data.get("active_constraints_before_presolve", it_data.get("n_active_constraints", 0)),
                        "active_nonzeros_before_presolve": it_data.get("active_nonzeros_before_presolve", it_data.get("n_active_nonzeros", 0)),
                        "presolved_variables": it_data.get("presolved_variables"),
                        "presolved_constraints": it_data.get("presolved_constraints"),
                        "presolved_nonzeros": it_data.get("presolved_nonzeros"),
                        "solver_backend": solver_backend,
                    })

                last = hist[-1] if hist else {}
                p_vars = last.get("presolved_variables")
                p_cons = last.get("presolved_constraints")
                p_coef = last.get("presolved_nonzeros")

                n_vars = p_vars if p_vars is not None else last.get("free_variables", last.get("original_variables"))
                n_cons = p_cons if p_cons is not None else (last.get("active_constraints_before_presolve") or last.get("original_constraints"))
                n_coef = p_coef if p_coef is not None else (last.get("active_nonzeros_before_presolve") or last.get("original_nonzeros"))

                if n_vars is not None: stats_by_age[age]["vars"].append(n_vars)
                if n_cons is not None: stats_by_age[age]["cons"].append(n_cons)
                if n_coef is not None: stats_by_age[age]["coefs"].append(n_coef)

                print(f"  n={n} age={age} seed={seed}: vars={n_vars}, cons={n_cons}, nonzeros={n_coef} ({rt:.1f}s)")
                detail_rows.append({
                    "n": n, "age": age, "seed": seed,
                    "n_variables": n_vars, "n_constraints": n_cons, "n_nonzeros": n_coef,
                    "free_variables": last.get("free_variables"),
                    "presolved_variables": p_vars,
                    "presolved_constraints": p_cons,
                    "presolved_nonzeros": p_coef,
                    "solver_backend": solver_backend,
                })

        p_ref = paper_table4_ref.get(n, {})
        summary_rows.append({
            "n": n,
            "Solver": solver_backend,
            "Our age=2 Ave.#Cons": int(np.mean(stats_by_age[2]["cons"])) if stats_by_age[2]["cons"] else "-",
            "Our age=2 Ave.#Var": int(np.mean(stats_by_age[2]["vars"])) if stats_by_age[2]["vars"] else "-",
            "Our age=2 Ave.#Coef": int(np.mean(stats_by_age[2]["coefs"])) if stats_by_age[2]["coefs"] else "-",
            "Paper age=2 Cons/Var/Coef": f"{p_ref.get('age2', '-')}",
            "Our age=5 Ave.#Cons": int(np.mean(stats_by_age[5]["cons"])) if stats_by_age[5]["cons"] else "-",
            "Our age=5 Ave.#Var": int(np.mean(stats_by_age[5]["vars"])) if stats_by_age[5]["vars"] else "-",
            "Our age=5 Ave.#Coef": int(np.mean(stats_by_age[5]["coefs"])) if stats_by_age[5]["coefs"] else "-",
            "Paper age=5 Cons/Var/Coef": f"{p_ref.get('age5', '-')}",
        })

    df_summary = pd.DataFrame(summary_rows)
    df_details = pd.DataFrame(detail_rows)
    df_iterations = pd.DataFrame(iteration_rows)

    df_summary.to_csv(OUT_DIR / "table4_reproduction.csv", index=False)
    df_details.to_csv(OUT_DIR / "table4_details.csv", index=False)
    df_iterations.to_csv(OUT_DIR / "table4_iterations_raw.csv", index=False)

    print("\n--- Completed Table 4 Reproduction Summary ---")
    print(df_summary.to_string(index=False))
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete Tables 3 and 4 reproduction")
    parser.add_argument("--protocol", choices=["paper", "smoke"], default="smoke")
    parser.add_argument("--backend", default=None)
    args = parser.parse_args()

    cfg = load_paper_protocol() if args.protocol == "paper" else load_smoke_protocol()
    backend = args.backend or cfg.solver.backend

    t_start = time.perf_counter()
    df_t3 = complete_table3(
        total_time=cfg.cmsa.total_time_limit_seconds,
        mip_time=cfg.cmsa.restricted_mip_time_limit_seconds,
        age_limit=cfg.cmsa.age_limit,
        solver_backend=backend,
        threads=cfg.solver.threads,
        mip_emphasis=cfg.solver.mip_emphasis,
    )
    df_t4 = run_table4(
        total_time=min(cfg.cmsa.total_time_limit_seconds, 30.0),
        mip_time=cfg.cmsa.restricted_mip_time_limit_seconds,
        solver_backend=backend,
        threads=cfg.solver.threads,
        mip_emphasis=cfg.solver.mip_emphasis,
        seeds=list(range(1, cfg.experiments.instances_per_size + 1)) if args.protocol == "paper" else [1, 2],
    )
    print(f"\nAll operations completed in {time.perf_counter() - t_start:.2f}s")
