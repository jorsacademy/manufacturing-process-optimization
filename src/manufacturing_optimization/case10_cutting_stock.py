"""Case 10: cutting-stock optimization with reusable remnants."""
from itertools import product
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def patterns(length, item_lengths):
    max_counts = [length // item_length for item_length in item_lengths]
    output = []
    for counts in product(*[range(max_count + 1) for max_count in max_counts]):
        used = sum(count * item_length for count, item_length in zip(counts, item_lengths))
        if used <= length and used > 0:
            output.append((counts, length - used))
    return output


def solve() -> dict:
    items = np.array([180, 260, 310], dtype=int)
    demand = np.array([8, 6, 5], dtype=int)
    stock_types = [
        (1000, 1.0, "new", None),
        (620, 0.25, "remnant_620", 2),
        (540, 0.20, "remnant_540", 2),
    ]

    all_patterns = []
    for length, cost, label, availability in stock_types:
        for counts, waste in patterns(length, items):
            all_patterns.append((np.array(counts), waste, cost, label, length, availability))

    n = len(all_patterns)
    cost = np.array([pattern[2] + 0.0008 * pattern[1] for pattern in all_patterns])
    production = np.column_stack([pattern[0] for pattern in all_patterns])

    rows = [-production]
    lower = [-np.inf * np.ones(len(demand))]
    upper = [-demand]

    for label in ("remnant_620", "remnant_540"):
        row = np.array([1.0 if pattern[3] == label else 0.0 for pattern in all_patterns])[None, :]
        availability = next(pattern[5] for pattern in all_patterns if pattern[3] == label)
        rows.append(row)
        lower.append(np.array([-np.inf]))
        upper.append(np.array([availability], dtype=float))

    matrix = np.vstack(rows)
    lo = np.concatenate(lower)
    hi = np.concatenate(upper)

    result = milp(
        c=cost,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.full(n, np.inf)),
        constraints=LinearConstraint(matrix, lo, hi),
    )
    if not result.success:
        raise RuntimeError(result.message)

    x = np.rint(result.x).astype(int)
    used = np.where(x > 0)[0]
    plan = [
        {
            "stock": all_patterns[i][3],
            "pattern": all_patterns[i][0].tolist(),
            "waste": int(all_patterns[i][1]),
            "count": int(x[i]),
        }
        for i in used
    ]
    produced = production @ x

    return {
        "plan": plan,
        "produced": produced,
        "demand": demand,
        "total_cost": float(cost @ x),
        "total_waste": float(sum(all_patterns[i][1] * x[i] for i in used)),
    }


if __name__ == "__main__":
    print(solve())
