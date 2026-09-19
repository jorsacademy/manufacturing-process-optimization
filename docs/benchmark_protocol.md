# Two-scale benchmark protocol

The fixture system separates **correctness evidence** from **scalability evidence**.

## Validation fixtures

Validation fixtures are deliberately small. Their purpose is to support:

- exact solution where practical;
- exhaustive enumeration or independent oracle checks;
- reconstruction of feasibility outside the solver;
- deterministic regression tests;
- sensitivity tests where a known small model is easier to interpret.

A validation result should report objective/KPIs, feasibility audit and solver status. For cases with an independent oracle, the optimized objective should also match that oracle within tolerance.

## Industrial fixtures

Industrial fixtures are deliberately too large for the naive exact methods used by some validation cases. Their purpose is to benchmark:

- model-build time;
- solve or policy-computation time;
- incumbent objective / optimality gap when available;
- memory-relevant dimensions such as binary-variable count;
- fallback rate for rolling-horizon methods;
- simulation replication uncertainty;
- solution feasibility and KPI stability under scale.

The intended industrial algorithm can differ from the small-instance oracle. Case 03 moves from enumeration to assignment MILP + dispatch; Case 07 from permutation enumeration to CP-SAT/LNS; Case 10 from complete pattern enumeration to column generation or a restricted pattern set.

## Reproducibility

Each fixture records:

- fixed seed;
- schema version;
- explicit units;
- dimensions;
- intended solver mode;
- SHA-256 fingerprint over the complete payload.

A changed fingerprint means the benchmark data changed and performance numbers should not be compared as if they came from the same fixture.

## CI policy

CI constructs and validates all 20 fixtures but does not solve every industrial fixture to optimality. This avoids turning correctness CI into an uncontrolled performance test. Full industrial campaigns should run separately and retain runtime, solver version, machine specification and result artifacts.
