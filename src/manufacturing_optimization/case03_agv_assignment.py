"""Case 03: AGV task assignment with battery and workload constraints."""
from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass(frozen=True)
class Instance:
    task_time: np.ndarray
    task_energy: np.ndarray
    battery: np.ndarray
    vehicle_speed_factor: np.ndarray


def demo_instance() -> Instance:
    return Instance(
        task_time=np.array([8, 6, 10, 7, 5, 9, 4], dtype=float),
        task_energy=np.array([7, 5, 9, 6, 4, 8, 3], dtype=float),
        battery=np.array([24, 25, 22], dtype=float),
        vehicle_speed_factor=np.array([1.0, 1.08, 0.94], dtype=float),
    )


def solve(instance: Instance | None = None) -> dict:
    ins = instance or demo_instance()
    n_tasks = len(ins.task_time)
    n_vehicles = len(ins.battery)
    best = None

    for assignment in product(range(n_vehicles), repeat=n_tasks):
        energy = np.zeros(n_vehicles)
        load = np.zeros(n_vehicles)
        for task, vehicle in enumerate(assignment):
            energy[vehicle] += ins.task_energy[task]
            load[vehicle] += ins.task_time[task] / ins.vehicle_speed_factor[vehicle]

        if np.any(energy > ins.battery + 1e-9):
            continue

        makespan = load.max()
        score = makespan + 0.15 * load.std()
        if best is None or score < best[0]:
            best = (score, assignment, energy.copy(), load.copy())

    if best is None:
        raise RuntimeError("No battery-feasible AGV assignment")

    _, assignment, energy, load = best
    return {
        "assignment": np.array(assignment, dtype=int),
        "vehicle_energy": energy,
        "vehicle_load": load,
        "makespan": float(load.max()),
        "battery_slack": ins.battery - energy,
    }


if __name__ == "__main__":
    print(solve())
