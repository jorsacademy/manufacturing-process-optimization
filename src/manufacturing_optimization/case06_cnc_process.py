"""Case 06: CNC cutting-parameter optimization with tool-life and quality economics."""
import numpy as np
from scipy.optimize import differential_evolution


def metrics(x: np.ndarray) -> dict:
    speed, feed, depth = x
    removal_rate = speed * feed * depth
    cycle_time = 9000.0 / removal_rate
    tool_life = 2.0e8 / (speed**2.4 * feed**0.9 * depth**0.25)
    roughness = 120.0 * feed**2
    power = 2.8 + 0.0045 * removal_rate
    energy = power * cycle_time / 60.0
    return {
        "speed": speed,
        "feed": feed,
        "depth": depth,
        "cycle_time": cycle_time,
        "tool_life": tool_life,
        "roughness": roughness,
        "energy": energy,
    }


def solve(seed: int = 42) -> dict:
    def objective(x):
        m = metrics(x)
        tool_cost = 18.0 * m["cycle_time"] / max(m["tool_life"], 1e-6)
        time_cost = 1.2 * m["cycle_time"]
        energy_cost = 0.22 * m["energy"]
        quality_penalty = 60.0 * max(0.0, m["roughness"] - 2.4) ** 2
        return time_cost + tool_cost + energy_cost + quality_penalty

    bounds = [(90, 260), (0.08, 0.28), (0.8, 2.5)]
    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        polish=True,
        maxiter=250,
    )
    output = metrics(result.x)
    output["objective"] = float(result.fun)
    output["success"] = bool(result.success)
    return output


if __name__ == "__main__":
    print(solve())
