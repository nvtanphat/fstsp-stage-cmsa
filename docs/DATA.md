# Data

## Public Agatz/Bouman/Schmidt benchmark

The public repository `pcbouman-eur/TSP-D-Instances` contains geometric TSP-with-drone instances, including the `restricted/maxradius` and `restricted/novisit` families referenced by the research lineage used in the supplied paper.

Download locally:

```bash
python scripts/download_agatz_data.py
```

Load one file:

```python
from fstsp.data.agatz_parser import load_geometric_instance
inst = load_geometric_instance("path/to/instance.txt")
```

### Parser semantics

The public format stores:

1. truck cost/time factor per unit Euclidean distance;
2. drone cost/time factor per unit Euclidean distance;
3. node count **including the depot**;
4. x/y/id location triplets;
5. optional `#MAXFLY` absolute drone-distance limit;
6. optional `#NOVISIT` node indices.

The internal model stores speeds, so factor `f` is represented as speed `1/f`; therefore `distance / speed == distance * f`.

For unrestricted files without `#MAXFLY`, the parser uses a finite 200%-of-maximum-pairwise-distance equivalent instead of an artificial `1e9`. This keeps the stage model's Big-M numerically reasonable while preserving an unrestricted two-leg drone operation.

## Generated research instances

The paper's exact 40 newly generated raw files are not provided in the supplied PDF. `generator.py` creates deterministic seeded geometric instances for controlled experiments. These are **reimplementation test instances**, not the authors' exact 40 files.

## Data integrity rule

Never overwrite downloaded raw benchmark files. Convert or filter them into another directory if a different representation is required.
