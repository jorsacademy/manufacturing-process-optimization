"""Case 08: CONWIP level selection using event-driven discrete-event simulation."""
from dataclasses import dataclass
import heapq
import numpy as np


@dataclass(frozen=True)
class Result:
    throughput: float
    mean_cycle_time: float
    mean_wip: float


def simulate(wip_cap: int, seed: int, horizon: float = 500.0) -> Result:
    rng = np.random.default_rng(seed)
    means = np.array([5.0, 7.0, 4.0])
    machine_free = np.zeros(3)
    queues = [[], [], []]
    events = []
    released = 0
    completed = 0
    in_system = 0
    entry = {}
    cycle = []
    area_wip = 0.0
    last_t = 0.0

    def release(now):
        nonlocal released, in_system
        while in_system < wip_cap:
            job = released
            released += 1
            in_system += 1
            entry[job] = now
            queues[0].append(job)

    def start_station(station, now):
        if machine_free[station] <= now + 1e-12 and queues[station]:
            job = queues[station].pop(0)
            processing = rng.lognormal(
                mean=np.log(means[station]) - 0.5 * 0.2**2,
                sigma=0.2,
            )
            finish = now + processing
            machine_free[station] = finish
            heapq.heappush(events, (finish, station, job))

    release(0.0)
    start_station(0, 0.0)

    while events:
        now, station, job = heapq.heappop(events)
        if now > horizon:
            break

        area_wip += in_system * (now - last_t)
        last_t = now

        if station < 2:
            queues[station + 1].append(job)
        else:
            completed += 1
            in_system -= 1
            cycle.append(now - entry.pop(job))
            release(now)

        start_station(station, now)
        if station < 2:
            start_station(station + 1, now)
        start_station(0, now)

    return Result(
        throughput=completed / horizon,
        mean_cycle_time=float(np.mean(cycle)) if cycle else np.inf,
        mean_wip=area_wip / max(last_t, 1e-9),
    )


def solve(caps=range(2, 11), seeds=range(10)) -> dict:
    rows = []
    for cap in caps:
        reps = [simulate(cap, seed) for seed in seeds]
        throughput = np.mean([r.throughput for r in reps])
        cycle_time = np.mean([r.mean_cycle_time for r in reps])
        mean_wip = np.mean([r.mean_wip for r in reps])
        score = cycle_time + 80.0 * max(0.0, 0.13 - throughput) + 0.4 * mean_wip
        rows.append((score, cap, throughput, cycle_time, mean_wip))

    score, cap, throughput, cycle_time, mean_wip = min(rows)
    return {
        "wip_cap": cap,
        "throughput": throughput,
        "mean_cycle_time": cycle_time,
        "mean_wip": mean_wip,
        "score": score,
    }


if __name__ == "__main__":
    print(solve())
