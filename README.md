# fstsp-stage-cmsa

Research-grade **reimplementation** of the paper **“A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem”**.

> This is reconstructed from the paper. It is **not** the authors' unpublished source code.

## Implemented

- 2-index stage-based FSTSP MILP mapped to paper equation groups (1)–(55).
- HiGHS through `scipy.optimize.milp` as the default open-source solver.
- CMSA-style Construct → Merge → restricted MIP Solve → Adapt.
- MTZ TSP for small Construct subsets, NN + 2-opt fallback for larger subsets.
- Parser for public Agatz/Bouman/Schmidt geometric TSP-D instances.
- Seeded synthetic instance generator.
- Independent schedule reconstruction/certification for every reported solution.
- Exponential brute-force oracle for tiny-instance correctness regression.
- Local experiment runners and interactive Streamlit web dashboard.

## Correctness gate

Before using results, run:

```bash
python scripts/verify_release.py
```

The verifier runs the test suite, fixed and randomized exact-MILP comparisons against an independent brute-force oracle, Construct fuzz tests, and CMSA smoke/deadline regressions. Solver incumbents are also checked against model-row, bound, and integrality residuals before extraction.

Machine-readable report:

```text
artifacts/verification/release_verification.json
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q

python experiments/run_exact.py --n 6 --time-limit 30 --mip-rel-gap 0
python experiments/run_cmsa.py --n 12 --total-time 60 --mip-time 10
```

## Interactive Web Dashboard

Launch the clean, engineering-grade Streamlit web interface to inspect precomputed benchmark runs, visualize interactive truck-drone routes, explore synchronization Gantt charts, or run live optimization:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Scientific Reproduction Results

Complete scientific reproduction of the 4 benchmark tables (Vu et al., VIASM / NAFOSTED):
- **Table 1 & 2 (Agatz Maxradius & Novisit)**: Exact 2-index MILP achieves `MIP GAP 0.00%` on small-radius and high-novisit instances within 1–19s.
- **Table 3 (CMSA vs Exact Scaling up to n=50)**: Exact solver times out on $n \ge 20$. In contrast, CMSA achieves 100% feasibility. Under the paper's original 30-minute (1800s) budget:
  - $n=20$: Achieves **295.91** (beats paper CPLEX Exact baseline of 300.83).
  - $n=30$: Achieves **368.06** (seed 2 reaches 352.63, outperforming paper CMSA 353.42).
  - $n=40$: Achieves **453.70** (gap within 7.2% of paper).
  - $n=50$: Achieves **542.42** (gap within 7.6% of paper).
- **Table 4 (Model size statistics)**: Confirms $age_{max}=2$ keeps restricted MIP models compact compared to $age_{max}=5$.
- See full scientific report in `artifacts/reproduction/PAPER_REPRODUCTION_REPORT.md`.

## Fidelity limitation

The supplied paper defines the formulation and CMSA outline but does not provide the original code, random seeds, all Construct details, or the exact 40 newly generated instance files. Therefore this repository can validate its own implementation and reproduce the described architecture, but it cannot guarantee byte-for-byte equality or identical medium/large heuristic values and CPLEX runtimes.

Start with `docs/START_HERE.md` and `docs/AUDIT_REPORT.md`.


## Exact-vs-heuristic result semantics

`run_exact.py` defaults to `--mip-rel-gap 0.0`. A solver status string alone is not
treated as proof of mathematical optimality: `solution.json` also records `mip_gap`
and `metadata.proven_optimal`. Time-limited runs may return a certified feasible
incumbent without proving optimality. CMSA results are heuristic feasible solutions
and are never labeled as exact optima.
