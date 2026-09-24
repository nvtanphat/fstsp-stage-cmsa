# Reproducibility and fidelity statement

## Directly supported by the supplied paper

The paper specifies:

- the stage-based FSTSP parameters and variables;
- objective `min d_E`;
- constraint groups (1)-(55);
- CMSA architecture and component types `x`, `phi`, `A`, `B`;
- `age_limit = 2` and restricted-MIP solve time `t_MIP = 15 s` in the reported experiment;
- CPLEX 22.11 and 8 threads for the authors' experiments;
- 2650 inherited benchmark instances plus 40 newly generated instances at reported sizes 20/30/40/50.

## Information not supplied by the paper

The supplied PDF does not provide:

- original source code;
- exact files for the 40 newly generated instances;
- random seeds;
- exact sampling distribution/policy used in Construct;
- all tie-breaking choices;
- all implementation-level solver settings;
- a machine-readable reference implementation.

Therefore byte-for-byte reproduction and guaranteed equality with the paper's heuristic numbers are not possible from the PDF alone.

## Reimplementation choices

- Default solver: HiGHS through `scipy.optimize.milp`.
- Big-M: conservative data-dependent bound in `stage_based.py`.
- Synthetic instances: seeded geometric generator.
- Construct: sampled truck-customer set, MTZ for small subsets, NN + 2-opt fallback, then certified consecutive-edge drone assignments.
- Restricted MIP: inactive `x/phi/A/B` components are fixed to zero.
- Safety incumbent: retain the best certified Construct solution because this HiGHS path does not provide a MIP-start hook.
- Every reported solution is independently schedule-certified.

## Exact model verification

Tiny instances are also solved by an independent exhaustive enumerator. The regression suite requires the stage-MILP objective to agree with this oracle within numerical tolerance.

This is evidence for implementation correctness on the tested tiny cases. It is not a proof for all possible instances.

## Public benchmark notation warning

The public `TSP-D-Instances` file header counts **nodes including the depot**. `FSTSPInstance.n` in this repository counts **customers only**. For paper/table reproduction, preserve the original file-name size label separately rather than assuming these two conventions are identical.

## Safe research claim

> We implemented and independently validated a reimplementation of the paper's 2-index stage-based FSTSP formulation and CMSA architecture using HiGHS, with documented reconstruction choices and reproducible experiment scripts.

Do not claim that this is the authors' original code or an exact reproduction of the CPLEX runtimes.
