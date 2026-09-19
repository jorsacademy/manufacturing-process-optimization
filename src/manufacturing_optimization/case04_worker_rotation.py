"""Case 04: skill-feasible worker rotation with ergonomic load balancing."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Instance:
    skill: np.ndarray
    ergonomic_risk: np.ndarray
    periods: int = 4


def demo_instance() -> Instance:
    return Instance(
        skill=np.array([
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=int),
        ergonomic_risk=np.array([5.0, 8.0, 4.0]),
        periods=4,
    )


def solve(instance: Instance | None = None) -> dict:
    ins = instance or demo_instance()
    workers, stations = ins.skill.shape
    periods = ins.periods
    nx = workers * stations * periods
    z = nx
    n = nx + 1

    c = np.zeros(n)
    c[z] = 1.0
    integrality = np.zeros(n, dtype=int)
    integrality[:nx] = 1
    lb = np.zeros(n)
    ub = np.ones(n)
    ub[z] = np.inf

    def ix(worker, station, period):
        return period * workers * stations + worker * stations + station

    rows, lo, hi = [], [], []

    for period in range(periods):
        for station in range(stations):
            row = np.zeros(n)
            for worker in range(workers):
                row[ix(worker, station, period)] = 1
            rows.append(row)
            lo.append(1)
            hi.append(1)

    for period in range(periods):
        for worker in range(workers):
            row = np.zeros(n)
            for station in range(stations):
                row[ix(worker, station, period)] = 1
            rows.append(row)
            lo.append(-np.inf)
            hi.append(1)

    for period in range(periods):
        for worker in range(workers):
            for station in range(stations):
                if ins.skill[worker, station] == 0:
                    row = np.zeros(n)
                    row[ix(worker, station, period)] = 1
                    rows.append(row)
                    lo.append(0)
                    hi.append(0)

    for worker in range(workers):
        row = np.zeros(n)
        for period in range(periods):
            for station in range(stations):
                row[ix(worker, station, period)] = ins.ergonomic_risk[station]
        row[z] = -1
        rows.append(row)
        lo.append(-np.inf)
        hi.append(0)

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(np.vstack(rows), np.array(lo), np.array(hi)),
    )
    if not result.success:
        raise RuntimeError(result.message)

    schedule = np.rint(result.x[:nx]).astype(int).reshape(periods, workers, stations)
    loads = np.array([
        sum(
            ins.ergonomic_risk[station] * schedule[period, worker, station]
            for period in range(periods)
            for station in range(stations)
        )
        for worker in range(workers)
    ])
    return {
        "schedule": schedule,
        "worker_risk": loads,
        "max_risk": float(loads.max()),
    }


if __name__ == "__main__":
    print(solve())
