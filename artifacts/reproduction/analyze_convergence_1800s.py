import json
from pathlib import Path
import pandas as pd

base = Path("artifacts/kaggle_download_v9")
records = []

for p in sorted(base.glob("table3_*_cmsa/solution.json")):
    folder_name = p.parent.name
    data = json.loads(p.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    history = meta.get("history", [])
    
    parts = folder_name.split("_")
    n = int(parts[1].replace("n", ""))
    seed = int(parts[2].replace("seed", ""))
    
    current_best = float("inf")
    for h in history:
        iter_num = h.get("iteration")
        elapsed = h.get("elapsed", 0.0)
        c_obj = h.get("constructed_objective")
        r_obj = h.get("restricted_objective")
        r_feas = h.get("restricted_feasible")
        candidates = []
        if c_obj is not None:
            candidates.append(c_obj)
        if r_feas and r_obj is not None:
            candidates.append(r_obj)
        
        if candidates:
            cand = min(candidates)
            if cand < current_best:
                current_best = cand
            
        records.append({
            "n": n,
            "seed": seed,
            "iteration": iter_num,
            "elapsed_seconds": round(elapsed, 1),
            "best_so_far": round(current_best, 2),
            "active_components": h.get("active_components")
        })

df = pd.DataFrame(records)
df.to_csv("artifacts/reproduction/convergence_history_1800s.csv", index=False)

print("=== 1800s FULL CONVERGENCE SUMMARY BY INSTANCE ===")
for (n, seed), group in df.groupby(["n", "seed"]):
    initial = group.iloc[0]["best_so_far"]
    final = group.iloc[-1]["best_so_far"]
    iters = len(group)
    last_time = group.iloc[-1]["elapsed_seconds"]
    pct_drop = ((initial - final) / initial) * 100
    print(f"n={n:2d} seed={seed} | {iters:3d} iters in {last_time/60:4.1f}m | Start: {initial:6.2f} -> End: {final:6.2f} (Drop: -{pct_drop:4.1f}%)")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sizes = [20, 30, 40, 50]
for idx, n_val in enumerate(sizes):
    ax = axes[idx // 2, idx % 2]
    sub = df[df["n"] == n_val]
    for s, grp in sub.groupby("seed"):
        ax.step(grp["elapsed_seconds"] / 60.0, grp["best_so_far"], label=f"Seed {s}", where="post")
    ax.set_title(f"CMSA Convergence (n = {n_val})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Elapsed Time (minutes)")
    ax.set_ylabel("Best Objective (Makespan)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

plt.tight_layout()
plt.savefig("artifacts/reproduction/figure_convergence_1800s.png", dpi=300)
plt.close()
print("Saved artifacts/reproduction/figure_convergence_1800s.png")
