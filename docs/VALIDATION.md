# Validation protocol

The repository uses three independent validation layers.

## Layer 1 — unit/integration/regression tests

Run:

```bash
pytest -q
```

The suite covers:

- deterministic instance generation;
- Agatz benchmark parsing and restriction semantics;
- finite unrestricted endurance mapping;
- CMSA age lifecycle;
- TSP/Construct integration;
- exact stage-model feasibility;
- solution schedule certification;
- stale-stage regression;
- positive-launch-time/start-depot regression;
- exact MILP vs independent brute-force optimum on tiny cases.

## Layer 2 — independent schedule reconstruction

`src/fstsp/evaluation/schedule.py` reconstructs the schedule only from:

- truck route;
- drone sorties;
- travel-time matrices;
- launch/recovery handling times;
- drone endurance.

It does not trust solver continuous-time variables. It checks coverage, ordering, one-drone non-overlap, Eq. (14)-style flight endurance, Eq. (33)-style rendezvous endurance, and completion time.

`solve_stage_model()` retains the raw solver objective in metadata but reports the independently reconstructed certified objective. If the discrete incumbent cannot be certified, it is returned as infeasible with certification issues in metadata.

## Layer 3 — brute-force oracle

`src/fstsp/evaluation/bruteforce.py` is an exponential tiny-instance oracle. It enumerates truck subsets/orders and compatible non-overlapping drone sorties, then evaluates each plan with the independent schedule reconstruction.

It is only for verification. It is not intended for practical solving.

## Release verification command

```bash
python scripts/verify_release.py
```

This generates:

```text
artifacts/verification/release_verification.json
```

A release should not be published if this command fails.

## Kaggle validation policy

The Kaggle script writes artifacts and then intentionally raises an error if any row has a non-empty validation field. Therefore a green Kaggle run means every emitted benchmark solution passed independent schedule certification.

This validates implementation consistency; it does not imply equality with the paper's CPLEX runtimes or unpublished generated instances.
