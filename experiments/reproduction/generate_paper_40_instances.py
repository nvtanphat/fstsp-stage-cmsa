"""Generate the 40 newly generated instances defined in Paper Section 4 & Appendix Table 7.

Sizes n in [20, 30, 40, 50], 10 instances each (seed 1 to 10).
Customer locations uniformly distributed in [0, 100] x [0, 100], depot at center (50, 50).
Truck speed = 1.0, Drone speed = 2.0, launch = 1.0, recovery = 1.0.
Max radius = 200% (unrestricted, endurance sufficient for any 2-leg sortie).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure src is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fstsp.data.generator import generate_uniform_instance


def main() -> None:
    out_dir = ROOT / "data" / "paper_40_instances"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [20, 30, 40, 50]
    num_instances_per_size = 10
    manifest = []

    for n in sizes:
        for seed in range(1, num_instances_per_size + 1):
            inst = generate_uniform_instance(
                n=n,
                seed=seed,
                width=100.0,
                truck_speed=1.0,
                drone_speed=2.0,
                drone_endurance=None,  # 200% max-pair practical default
                launch_time=1.0,
                recovery_time=1.0,
                novisit_fraction=0.0,
            )
            file_path = out_dir / f"instance_n{n}_seed{seed:02d}.json"
            inst.to_json(file_path)
            manifest.append({
                "name": inst.name,
                "n": n,
                "seed": seed,
                "file": file_path.name,
                "depot": inst.depot_coord.tolist(),
                "num_customers": inst.n,
                "drone_endurance": inst.drone_endurance,
            })
            print(f"Generated {inst.name} -> {file_path.name}")

    manifest_file = out_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nGenerated all 40 instances in {out_dir}")


if __name__ == "__main__":
    main()
