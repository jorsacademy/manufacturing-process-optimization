"""Case 05: inspection/rework policy optimization as a finite MDP."""
import numpy as np

STATES = ("stable", "drifting", "out_of_control")
ACTIONS = ("light", "standard", "intensive")


def model():
    transition = np.array([
        [[0.88, 0.10, 0.02], [0.93, 0.06, 0.01], [0.97, 0.025, 0.005]],
        [[0.12, 0.70, 0.18], [0.28, 0.63, 0.09], [0.55, 0.40, 0.05]],
        [[0.02, 0.18, 0.80], [0.08, 0.37, 0.55], [0.45, 0.45, 0.10]],
    ])
    cost = np.array([
        [8, 16, 31],
        [55, 38, 34],
        [170, 105, 58],
    ], dtype=float)
    return transition, cost


def solve(gamma: float = 0.96, tol: float = 1e-10) -> dict:
    transition, cost = model()
    value = np.zeros(len(STATES))

    for _ in range(10000):
        q = cost + gamma * np.einsum("sak,k->sa", transition, value)
        new_value = q.min(axis=1)
        if np.max(np.abs(new_value - value)) < tol:
            value = new_value
            break
        value = new_value

    q = cost + gamma * np.einsum("sak,k->sa", transition, value)
    policy = q.argmin(axis=1)
    return {
        "value": value,
        "policy": {state: ACTIONS[int(action)] for state, action in zip(STATES, policy)},
        "bellman_residual": float(np.max(np.abs(value - q.min(axis=1)))),
    }


if __name__ == "__main__":
    print(solve())
