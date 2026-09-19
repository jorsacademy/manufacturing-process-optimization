"""Case 01: integrated production assignment and preventive-maintenance planning."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Instance:
    processing: np.ndarray
    maintenance_duration: np.ndarray
    maintenance_cost: np.ndarray
    failure_risk_cost: np.ndarray


def demo_instance() -> Instance:
    return Instance(
        processing=np.array([
            [4.0, 5.0], [3.0, 4.5], [5.5, 4.0],
            [2.5, 3.0], [4.5, 3.5], [3.5, 4.0],
        ]),
        maintenance_duration=np.array([2.0, 1.5]),
        maintenance_cost=np.array([140.0, 120.0]),
        failure_risk_cost=np.array([520.0, 430.0]),
    )


def solve(instance: Instance | None = None) -> dict:
    ins = instance or demo_instance()
    j, m = ins.processing.shape
    n_x = j * m
    pm0 = n_x
    cmax_idx = n_x + m
    n = cmax_idx + 1

    c = np.zeros(n)
    c[pm0:pm0 + m] = ins.maintenance_cost - ins.failure_risk_cost
    c[cmax_idx] = 100.0

    integrality = np.zeros(n, dtype=int)
    integrality[:n_x + m] = 1
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    ub[:n_x + m] = 1.0

    rows, lo, hi = [], [], []
    for job in range(j):
        row = np.zeros(n)
        for machine in range(m):
            row[job * m + machine] = 1
        rows.append(row)
        lo.append(1)
        hi.append(1)

    for machine in range(m):
        row = np.zeros(n)
        for job in range(j):
            row[job * m + machine] = ins.processing[job, machine]
        row[pm0 + machine] = ins.maintenance_duration[machine]
        row[cmax_idx] = -1
        rows.append(row)
        lo.append(-np.inf)
        hi.append(0)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(np.vstack(rows), np.array(lo), np.array(hi)),
        options={"time_limit": 10.0},
    )
    if not result.success:
        raise RuntimeError(result.message)

    x = result.x[:n_x].reshape(j, m)
    pm = np.rint(result.x[pm0:pm0 + m]).astype(int)
    assignment = np.argmax(x, axis=1)
    loads = np.array([
        sum(ins.processing[job, machine] for job in range(j) if assignment[job] == machine)
        + pm[machine] * ins.maintenance_duration[machine]
        for machine in range(m)
    ])
    total_cost = result.fun + float(ins.failure_risk_cost.sum())
    return {
        "assignment": assignment,
        "preventive_maintenance": pm,
        "machine_loads": loads,
        "makespan": float(loads.max()),
        "total_cost": float(total_cost),
    }


if __name__ == "__main__":
    print(solve())
