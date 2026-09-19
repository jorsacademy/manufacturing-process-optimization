"""Case 07: two-press scheduling with mold compatibility and changeovers."""
from dataclasses import dataclass
from itertools import permutations, product
import numpy as np


@dataclass(frozen=True)
class Job:
    duration: tuple[float, float]
    mold: str
    compatible: tuple[int, ...]


def demo_jobs():
    return [
        Job((4, 5), "A", (0, 1)),
        Job((3, 4), "A", (0, 1)),
        Job((6, 5), "B", (0, 1)),
        Job((4, 3), "C", (1,)),
        Job((5, 6), "B", (0, 1)),
        Job((2, 3), "C", (0, 1)),
    ]


def sequence_time(jobs, sequence, machine, setup=1.5):
    total = 0.0
    previous = None
    for job in sequence:
        if previous is not None and jobs[previous].mold != jobs[job].mold:
            total += setup
        total += jobs[job].duration[machine]
        previous = job
    return total


def solve(jobs=None) -> dict:
    jobs = jobs or demo_jobs()
    choices = [job.compatible for job in jobs]
    best = None

    for assignment in product(*choices):
        by_machine = [[j for j, a in enumerate(assignment) if a == machine] for machine in (0, 1)]
        seqs0 = permutations(by_machine[0]) if by_machine[0] else [()]
        for sequence0 in seqs0:
            time0 = sequence_time(jobs, sequence0, 0)
            seqs1 = permutations(by_machine[1]) if by_machine[1] else [()]
            for sequence1 in seqs1:
                time1 = sequence_time(jobs, sequence1, 1)
                objective = max(time0, time1) + 0.05 * (time0 + time1)
                if best is None or objective < best[0]:
                    best = (objective, sequence0, sequence1, time0, time1)

    _, sequence0, sequence1, time0, time1 = best
    return {
        "press_0": sequence0,
        "press_1": sequence1,
        "loads": np.array([time0, time1]),
        "makespan": float(max(time0, time1)),
    }


if __name__ == "__main__":
    print(solve())
