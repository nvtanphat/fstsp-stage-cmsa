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
        
        cand = r_obj if (r_feas and r_obj is not None) else c_obj
        if cand is not None and cand < current_best:
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
