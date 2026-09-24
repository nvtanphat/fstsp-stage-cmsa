from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fstsp.algorithms.cmsa.algorithm import solve_cmsa
from fstsp.algorithms.cmsa.construct import construct_solution
from fstsp.data.generator import generate_uniform_instance
from fstsp.evaluation.bruteforce import brute_force_optimum
from fstsp.evaluation.validator import validate_solution
from fstsp.formulation.stage_based import solve_stage_model


def run_cmd(cmd: list[str], cwd: Path = ROOT, timeout: int = 120) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - started,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> None:
    report: dict = {"checks": {}}

    pytest_result = run_cmd([sys.executable, "-m", "pytest", "-q"], timeout=400)
    report["checks"]["pytest"] = pytest_result
    if pytest_result["returncode"] != 0:
        raise SystemExit("pytest failed")

    # Independent exact-vs-enumeration oracle. Cover zero/positive handling,
    # finite endurance, and NOVISIT restrictions so correctness is not only
    # demonstrated on the easiest benchmark semantics.
    oracle_rows = []
    max_abs_diff = 0.0
    oracle_cases = [
        (2, 1, 0.0, 0.0, 120.0, 0.0),
        (2, 4, 1.0, 1.0, 120.0, 0.0),
        (3, 2, 0.0, 1.0, 60.0, 0.0),
        (3, 6, 2.0, 3.0, 80.0, 0.0),
        (4, 1, 1.0, 1.0, 80.0, 0.25),
        (4, 5, 0.0, 0.0, 50.0, 0.25),
    ]
    for n, seed, launch, recovery, endurance, novisit in oracle_cases:
        inst = generate_uniform_instance(
            n,
            seed=seed,
            width=30.0,
            launch_time=launch,
            recovery_time=recovery,
            drone_endurance=endurance,
            novisit_fraction=novisit,
        )
        oracle = brute_force_optimum(inst, max_customers=4)
        exact = solve_stage_model(inst, time_limit=30.0, mip_rel_gap=0.0)
        issues = validate_solution(inst, exact)
        if not oracle.feasible or not exact.feasible or issues:
            raise SystemExit(f"exact/oracle certification failed n={n}, seed={seed}: {issues}")
        diff = abs(float(exact.objective) - float(oracle.objective))
        max_abs_diff = max(max_abs_diff, diff)
        oracle_rows.append(
            {
                "n": n,
                "seed": seed,
                "launch": launch,
                "recovery": recovery,
                "endurance": endurance,
                "novisit_fraction": novisit,
                "oracle": oracle.objective,
                "milp": exact.objective,
                "abs_diff": diff,
            }
        )
    report["checks"]["exact_vs_bruteforce"] = {
        "cases": oracle_rows,
        "max_abs_diff": max_abs_diff,
        "passed": max_abs_diff <= 2e-5,
    }
    if max_abs_diff > 2e-5:
        raise SystemExit(f"exact MILP differs from brute-force oracle by {max_abs_diff}")


    # Randomized independent-oracle stress over varied scales, speeds, handling times,
    # endurance and NOVISIT masks. This is intentionally separate from pytest cases.
    random_rows = []
    random_max_diff = 0.0
    for case in range(40):
        rng = np.random.default_rng(70000 + case)
        n = 2 + (case % 3)  # 2..4 customers; exhaustive oracle remains tractable
        scale = 10.0 ** float(rng.uniform(-1.0, 2.0))
        from fstsp.domain.instance import FSTSPInstance
        inst = FSTSPInstance(
            name=f"release_fuzz_{case}",
            coords=rng.uniform(-scale, scale, size=(n, 2)),
            depot_coord=np.array([0.0, 0.0]),
            truck_speed=float(10.0 ** rng.uniform(-0.4, 0.4)),
            drone_speed=float(10.0 ** rng.uniform(-0.2, 0.6)),
            drone_endurance=float(10.0 ** rng.uniform(0.0, 2.2)),
            launch_time=float(rng.choice([0.0, 0.1, 1.0, 2.0])),
            recovery_time=float(rng.choice([0.0, 0.1, 1.0, 2.0])),
            drone_allowed=(rng.random(n) > 0.25),
        )
        oracle = brute_force_optimum(inst, max_customers=4)
        exact = solve_stage_model(inst, time_limit=20.0, mip_rel_gap=0.0)
        issues = validate_solution(inst, exact) if exact.feasible else []
        if oracle.feasible != exact.feasible or issues:
            raise SystemExit(
                f"random exact/oracle mismatch case={case}: exact={exact.status}, issues={issues}"
            )
        diff = 0.0
        if oracle.feasible:
            diff = abs(float(exact.objective) - float(oracle.objective))
            random_max_diff = max(random_max_diff, diff)
            if diff > 2e-5:
                raise SystemExit(f"random exact/oracle objective mismatch case={case}: {diff}")
        random_rows.append({
            "case": case,
            "n": n,
            "abs_diff": diff,
            "max_row_violation": exact.metadata.get("max_row_violation"),
            "max_integrality_violation": exact.metadata.get("max_integrality_violation"),
        })
    report["checks"]["randomized_exact_vs_bruteforce"] = {
        "cases": len(random_rows),
        "max_abs_diff": random_max_diff,
        "passed": True,
    }

    # Regression fuzz for the former stale-sortie construction bug.
    fuzz_cases = 0
    for n in [5, 8, 10, 12, 20]:
        for seed in range(20):
            inst = generate_uniform_instance(
                n,
                seed=seed,
                width=100.0,
                drone_endurance=30.0,
                launch_time=1.0,
                recovery_time=1.0,
                novisit_fraction=0.2,
            )
            sol = construct_solution(
                inst,
                np.random.default_rng(seed),
                truck_sample_ratio=0.5,
                exact_tsp_threshold=8,
            )
            issues = validate_solution(inst, sol)
            if issues:
                raise SystemExit(f"construct fuzz failed n={n}, seed={seed}: {issues}")
            fuzz_cases += 1
    report["checks"]["construct_fuzz"] = {"cases": fuzz_cases, "passed": True}

    # Short CMSA end-to-end certification.
    cmsa_rows = []
    for n, seed in [(6, 1), (8, 2), (10, 3)]:
        inst = generate_uniform_instance(
            n,
            seed=seed,
            width=100.0,
            drone_endurance=80.0,
            launch_time=1.0,
            recovery_time=1.0,
        )
        sol = solve_cmsa(inst, total_time=1.0, mip_time=0.25, seed=seed)
        issues = validate_solution(inst, sol)
        if not sol.feasible or issues:
            raise SystemExit(f"CMSA certification failed n={n}, seed={seed}: {issues}")
        cmsa_rows.append(
            {
                "n": n,
                "seed": seed,
                "objective": sol.objective,
                "runtime": sol.runtime,
                "iterations": sol.metadata.get("iterations"),
            }
        )
    report["checks"]["cmsa_smoke"] = {"cases": cmsa_rows, "passed": True}


    # Regression for v0.3 wall-clock overrun: model assembly used to occur outside
    # the CMSA deadline and could overshoot a sub-second budget by >1 second at n=20.
    budget_inst = generate_uniform_instance(
        20, seed=77, width=100.0, drone_endurance=50.0,
        launch_time=1.0, recovery_time=1.0, novisit_fraction=0.2
    )
    budget = 0.35
    budget_sol = solve_cmsa(
        budget_inst, total_time=budget, mip_time=0.10, construct_tsp_time=0.05, seed=77
    )
    budget_issues = validate_solution(budget_inst, budget_sol)
    if not budget_sol.feasible or budget_issues or budget_sol.runtime > budget + 0.20:
        raise SystemExit(
            f"CMSA budget regression failed: runtime={budget_sol.runtime}, issues={budget_issues}"
        )
    report["checks"]["cmsa_wallclock_budget"] = {
        "budget_seconds": budget,
        "runtime_seconds": budget_sol.runtime,
        "overrun_seconds": budget_sol.metadata.get("budget_overrun_seconds"),
        "passed": True,
    }

    # Verify the exact folder that Kaggle CLI will upload can run standalone (if available).
    prep_script = ROOT / "scripts" / "prepare_kaggle_kernel.py"
    if prep_script.exists():
        prep = run_cmd(
            [
                sys.executable,
                str(prep_script),
                "--username",
                "verification-user",
                "--private",
                "--sizes",
                "4",
                "--seeds",
                "1",
                "--total-time",
                "0.75",
                "--mip-time",
                "0.2",
            ]
        )
        if prep["returncode"] != 0:
            raise SystemExit("Kaggle bundle preparation failed")
        kernel_dir = ROOT / "dist" / "kaggle_kernel"
        kernel_run = run_cmd([sys.executable, "run_experiment.py"], cwd=kernel_dir, timeout=120)
        report["checks"]["kaggle_bundle_local_smoke"] = {
            "prepare": prep,
            "run": kernel_run,
            "passed": kernel_run["returncode"] == 0,
        }
        if kernel_run["returncode"] != 0:
            raise SystemExit("Kaggle bundle local smoke failed")

    report["passed"] = True
    report["note"] = (
        "Local verification passed. A real Kaggle cloud push still requires the user's "
        "Kaggle authentication and is not claimed here."
    )
    out = ROOT / "artifacts" / "verification" / "release_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": True, "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
