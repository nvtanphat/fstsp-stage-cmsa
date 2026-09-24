# Architecture

```text
Data / instance
      |
      v
FSTSPInstance -----------------------------+
      |                                    |
      v                                    v
2-index stage MILP                   CMSA Construct
      |                                    |
      |                              solution components
      |                              x, phi, A, B
      |                                    |
      +---------------< Merge / Age >------+ 
                         |
                         v
                  Restricted stage MILP
                         |
                         v
                     Best solution
                         |
                 +-------+--------+
                 v                v
             metrics          visualization
```

## Package responsibilities

- `domain/`: immutable problem representation and solution DTOs.
- `data/`: generators and parsers; no optimization logic.
- `formulation/`: stage-based MILP only.
- `algorithms/tsp/`: truck-tour subroutines used by Construct.
- `algorithms/cmsa/`: CMSA state machine and component aging.
- `evaluation/`: feasibility checks and metrics.
- `visualization/`: figures only.
- `experiments/`: protocols/entry points.
- `notebooks/eda/`: EDA and post-hoc result analysis only.

## Solver choice

The paper reports CPLEX 22.11. This reimplementation defaults to SciPy's `milp` interface backed by HiGHS because it is open-source and works in a standard Kaggle Python environment without a commercial license. Therefore objective values can be compared on the same instance, but raw runtime should not be treated as solver-equivalent to the paper.
