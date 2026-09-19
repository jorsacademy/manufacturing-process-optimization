# Manufacturing Process Optimization

Industrial Engineering / Operations Research monorepo with 10 manufacturing decision systems implemented in Python. Version 0.2 moves the project from isolated solver demos to auditable case studies with operational baselines, independent feasibility checks, KPI comparisons and stress/sensitivity analysis.

## Industrial case portfolio

| Case | Shop-floor decision | Method | Industrial additions |
|---|---|---|---|
| 01 | Production assignment + preventive maintenance | MILP | heterogeneous machines, failure-risk economics, PM downtime, overtime, risk sensitivity |
| 02 | Milk-run / line-side replenishment | MILP | safety stock, storage limits, multiple trips, emergency supply, vehicle-capacity sensitivity |
| 03 | AGV/AMR task allocation | exact combinatorial optimization | eligibility, battery reserve, due times, priorities, workload balance |
| 04 | Worker rotation | MILP | skill matrix, fatigue-weighted ergonomic exposure, exposure caps, high-risk rotation |
| 05 | Quality inspection policy | discounted MDP | inspection cost, escape/rework economics, exact policy evaluation, exhaustive cross-check |
| 06 | CNC process parameters | nonlinear global optimization | tool life, roughness, spindle power, energy, hardness stress test |
| 07 | Mold/tooling-aware press scheduling | exact scheduling | compatibility, sequence-dependent setup matrix, due dates, weighted tardiness |
| 08 | CONWIP control | discrete-event simulation optimization | warm-up, stochastic processing, breakdown/repair, replication uncertainty |
| 09 | Energy-aware production scheduling | time-indexed MILP | TOU tariff, peak-demand charge, weighted tardiness, maintenance blackout |
| 10 | Cutting stock + remnants | integer optimization | kerf, finite remnant stock, trim loss, overproduction accounting |

## Engineering standard used in every case

Each case contains four layers:

1. Decision model — explicit operational variables and constraints.
2. Baseline policy — a transparent heuristic or incumbent-style policy.
3. Independent audit — feasibility is reconstructed outside the solver result.
4. Decision evidence — optimized vs baseline KPI comparison; selected cases also expose sensitivity/stress functions.

The bundled data are deterministic or seeded synthetic fixtures. They are designed to exercise the model logic and are not presented as measurements from a particular plant.

## Install and validate

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    pytest

## Run the complete benchmark panel

    python -m manufacturing_optimization.benchmarks

Run individual cases:

    python -m manufacturing_optimization.case01_production_maintenance
    python -m manufacturing_optimization.case06_cnc_process
    python -m manufacturing_optimization.case08_conwip

## Repository structure

    src/manufacturing_optimization/
      validation.py
      benchmarks.py
      case01_production_maintenance.py
      ...
      case10_cutting_stock.py
    docs/
      industrialization.md
      data_contracts.md
      case01.md ... case10.md
    tests/
      test_cases.py
      test_industrial_cases.py

## Scope and interpretation

The project is a portfolio/reference implementation, not a generic APS/MES package. Exact enumeration is deliberately used where the bundled instance is small enough to make the result independently checkable. Larger plant deployments would replace those exact small-instance engines with decomposition, CP-SAT/MILP, metaheuristics or rolling-horizon control while retaining the same data contracts, audits and KPI definitions.
