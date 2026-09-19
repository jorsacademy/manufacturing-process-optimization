"""Case 09: time-indexed energy-aware single-machine scheduling MILP."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def solve() -> dict:
    processing = np.array([3, 2, 4, 2], dtype=int)
    power = np.array([5.0, 7.0, 4.0, 6.0])
    due = np.array([5, 7, 9, 10], dtype=float)
    tariff = np.array([0.12, 0.11, 0.10, 0.32, 0.35, 0.30, 0.16, 0.14, 0.13, 0.12, 0.11, 0.10])

    horizon = len(tariff)
    jobs = len(processing)
    starts = [list(range(0, horizon - processing[job] + 1)) for job in range(jobs)]

    offsets = []
    cursor = 0
    for candidates in starts:
        offsets.append(cursor)
        cursor += len(candidates)

    tard0 = cursor
    peak_idx = tard0 + jobs
    n = peak_idx + 1

    c = np.zeros(n)
    for job in range(jobs):
        for local, start in enumerate(starts[job]):
            c[offsets[job] + local] = power[job] * tariff[start:start + processing[job]].sum()
    c[tard0:tard0 + jobs] = 8.0
    c[peak_idx] = 0.8

    integrality = np.zeros(n, dtype=int)
    integrality[:tard0] = 1
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    ub[:tard0] = 1

    rows, lo, hi = [], [], []

    for job in range(jobs):
        row = np.zeros(n)
        row[offsets[job]:offsets[job] + len(starts[job])] = 1
        rows.append(row)
        lo.append(1)
        hi.append(1)

    for time in range(horizon):
        occupancy = np.zeros(n)
        power_row = np.zeros(n)
        for job in range(jobs):
            for local, start in enumerate(starts[job]):
                if start <= time < start + processing[job]:
                    idx = offsets[job] + local
                    occupancy[idx] = 1
                    power_row[idx] = power[job]
        rows.append(occupancy)
        lo.append(-np.inf)
        hi.append(1)

        power_row[peak_idx] = -1
        rows.append(power_row)
        lo.append(-np.inf)
        hi.append(0)

    for job in range(jobs):
        row = np.zeros(n)
        for local, start in enumerate(starts[job]):
            row[offsets[job] + local] = start + processing[job]
        row[tard0 + job] = -1
        rows.append(row)
        lo.append(-np.inf)
        hi.append(due[job])

    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=LinearConstraint(np.vstack(rows), np.array(lo), np.array(hi)),
    )
    if not result.success:
        raise RuntimeError(result.message)

    chosen = []
    for job in range(jobs):
        values = result.x[offsets[job]:offsets[job] + len(starts[job])]
        chosen.append(starts[job][int(np.argmax(values))])

    return {
        "start_times": np.array(chosen),
        "tardiness": result.x[tard0:tard0 + jobs],
        "peak_power": float(result.x[peak_idx]),
        "total_cost": float(result.fun),
    }


if __name__ == "__main__":
    print(solve())
