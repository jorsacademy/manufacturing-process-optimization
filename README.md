# Manufacturing Process Optimization

A focused Industrial Engineering / Operations Research monorepo containing ten independent manufacturing decision problems implemented in Python.

The repository is intentionally broader than a single scheduling demo. Each case targets a different shop-floor decision layer: production/maintenance coordination, line feeding, AGV assignment, workforce ergonomics, quality control, machining parameters, tooling, WIP control, energy-aware scheduling, and material yield.

## Case studies

| # | Case | Method | Primary decision |
|---|---|---|---|
| 01 | Integrated production + maintenance | MILP | job assignment and preventive maintenance |
| 02 | Milk-run line feeding | MILP | deliveries, trips, line-side inventory |
| 03 | AGV/AMR task assignment | exact enumeration | task-to-vehicle allocation under battery limits |
| 04 | Ergonomic worker rotation | MILP | worker-station-period assignment |
| 05 | Quality inspection policy | discounted MDP | inspection intensity by process state |
| 06 | CNC process parameters | nonlinear global optimization | speed, feed, depth of cut |
| 07 | Tooling / mold scheduling | exact combinatorial search | press assignment and mold-aware sequencing |
| 08 | CONWIP control | discrete-event simulation optimization | WIP cap |
| 09 | Energy-aware production scheduling | time-indexed MILP | job start times under TOU tariffs |
| 10 | Cutting stock with remnants | integer optimization | cutting patterns and remnant consumption |

## Design principles

- Synthetic data are explicit and reproducible; no values are presented as plant measurements.
- Exact or independently checkable methods are preferred on the bundled small instances.
- Every case has a smoke/regression test.
- The cases deliberately use different OR paradigms instead of cloning one scheduling formulation ten times.

## Installation

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    pytest

## Run a case

    python -m manufacturing_optimization.case01_production_maintenance
    python -m manufacturing_optimization.case05_quality_mdp
    python -m manufacturing_optimization.case08_conwip

## Scope

These are compact, auditable reference implementations for Industrial Engineering portfolios and experimentation. The bundled instances are deliberately small enough to validate behavior quickly. They are not presented as drop-in MES/APS software or as calibrated models of a specific factory.
