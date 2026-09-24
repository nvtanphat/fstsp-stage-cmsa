# START HERE

## Scope

Research reimplementation of:

**A 2-index Stage-based Formulation and a Construct-Merge-Solve & Adapt Algorithm for the Flying Sidekick Traveling Salesman Problem**.

The repository is not claimed to be the authors' original code.

## First command to run

```bash
python scripts/verify_release.py
```

This runs the automated correctness gate and writes:

```text
artifacts/verification/release_verification.json
```

Then run a small exact experiment:

```bash
python experiments/run_exact.py --n 6 --time-limit 30
```

and a CMSA experiment:

```bash
python experiments/run_cmsa.py --n 12 --total-time 60 --mip-time 10 --age-limit 2
```

## Correctness design

A solver status alone is not trusted. Every returned discrete truck/drone plan is reconstructed into an independent schedule. Exact MILP results are also regression-tested against a brute-force oracle on tiny instances.

Read in this order:

1. `docs/HUONG_DAN_VI.md`
2. `docs/PAPER_MAPPING.md`
3. `docs/AUDIT_REPORT.md`
4. `docs/VALIDATION.md`
5. `docs/KAGGLE_CLI.md`
6. `docs/REPRODUCIBILITY.md`

## Kaggle

Kaggle deployment is a CPU script kernel. Core algorithms remain in `src/`; notebooks are for EDA only.
