"""Case 02: line-side replenishment with milk-run capacity and inventory balance."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Instance:
    demand: np.ndarray
    initial_inventory: np.ndarray
    vehicle_capacity: float = 30.0
    trip_cost: float = 55.0
    holding_cost: float = 0.8
    shortage_cost: float = 60.0


def demo_instance() -> Instance:
    return Instance(
        demand=np.array([[8, 10, 7, 11], [6, 9, 8, 7], [5, 6, 9, 8]], dtype=float),
        initial_inventory=np.array([8, 6, 6], dtype=float),
    )


def solve(instance: Instance | None = None) -> dict:
    ins = instance or demo_instance()
    s, t = ins.demand.shape
    nq = s * t
    ni = s * t
    ns = s * t
    q0, i0, sh0, y0 = 0, nq, nq + ni, nq + ni + ns
    n = y0 + t

    c = np.zeros(n)
    c[i0:i0 + ni] = ins.holding_cost
    c[sh0:sh0 + ns] = ins.shortage_cost
    c[y0:y0 + t] = ins.trip_cost

    integrality = np.zeros(n, dtype=int)
    integrality[y0:] = 1
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    ub[y0:] = 1

    rows, lo, hi = [], [], []

    def idx(base, station, period):
        return base + station * t + period

    for station in range(s):
        for period in range(t):
            row = np.zeros(n)
            row[idx(i0, station, period)] = 1
            row[idx(q0, station, period)] = -1
            row[idx(sh0, station, period)] = -1
            rhs = -ins.demand[station, period]
            if period == 0:
                rhs += ins.initial_inventory[station]
            else:
                row[idx(i0, station, period - 1)] = -1
            rows.append(row)
            lo.append(rhs)
            hi.append(rhs)

    for period in range(t):
        row = np.zeros(n)
        for station in range(s):
            row[idx(q0, station, period)] = 1
        row[y0 + period] = -ins.vehicle_capacity
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

    q = result.x[q0:q0 + nq].reshape(s, t)
    inventory = result.x[i0:i0 + ni].reshape(s, t)
    shortage = result.x[sh0:sh0 + ns].reshape(s, t)
    trips = np.rint(result.x[y0:y0 + t]).astype(int)
    return {
        "deliveries": q,
        "inventory": inventory,
        "shortage": shortage,
        "trips": trips,
        "total_cost": float(result.fun),
        "service_level": float(1 - shortage.sum() / ins.demand.sum()),
    }


if __name__ == "__main__":
    print(solve())
