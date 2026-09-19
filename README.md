# Manufacturing Process Optimization

A portfolio-style monorepo containing **10 independent Industrial Engineering / Operations Research case studies** for manufacturing systems. The emphasis is on decision models that have a recognizable shop-floor interpretation rather than generic optimization demos.

All bundled data are synthetic and are intended for reproducible experimentation. No repository result should be interpreted as a claim about a real plant.

## Case studies

| # | Case | Decision layer | Method | Primary KPI |
|---:|---|---|---|---|
| 01 | Integrated production + maintenance | production / maintenance | exact small-instance scheduling | weighted tardiness + setup + maintenance |
| 02 | Plant milk-run / supermarket feeding | internal logistics | MILP | replenishment + shortage cost |
| 03 | AGV/AMR production feeding | material handling | MILP | assignment + lateness + battery risk |
| 04 | Ergonomic worker rotation | workforce / human factors | MILP | staffing + ergonomic exposure |
| 05 | Quality inspection + rework policy | quality | discounted MDP | long-run quality cost |
| 06 | CNC process-parameter optimization | machining | nonlinear global optimization | machining + tool + quality + energy cost |
| 07 | Tooling / die constrained scheduling | tooling + scheduling | exact enumeration | makespan + changeover cost |
| 08 | CONWIP / Kanban WIP control | production control | discrete-event simulation optimization | throughput / cycle time / WIP |
| 09 | Energy-aware production scheduling | production + energy | MILP | energy + peak demand + setup + inventory |
| 10 | Cutting stock with remnant reuse | material yield | pattern-generation MILP | purchased stock + waste |

The cases intentionally use different mathematical structures. They are not ten cosmetic variations of the same scheduling model.

## Repository structure

```text
src/manufacturing_process_optimization/
    case01_production_maintenance.py
    case02_milkrun.py
    case03_agv_assignment.py
    case04_worker_rotation.py
    case05_quality_mdp.py
    case06_cnc_parameters.py
    case07_tooling_scheduling.py
    case08_conwip.py
    case09_energy_scheduling.py
    case10_cutting_stock.py
examples/
    case01.py ... case10.py
tests/
    test_case01.py ... test_case10.py
docs/
    MODELING_NOTES.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
```

## Run

Each example can be run independently:

```bash
python examples/case01.py
python examples/case08.py
python examples/case10.py
```

Run the full regression suite with:

```bash
pytest
```

## Modeling principles

1. Decision variables have an operational interpretation.
2. Constraints are enforced by the model and independently checked where practical.
3. Every case includes a transparent deterministic or policy baseline.
4. Stochastic experiments use fixed random seeds and, where useful, common random numbers.
5. Solver output is not treated as self-validating; feasibility and KPI reconstruction are explicit.
6. Synthetic assumptions are stated as synthetic instead of being presented as field measurements.

See [`docs/MODELING_NOTES.md`](docs/MODELING_NOTES.md) for the scope and limitations of each case.
