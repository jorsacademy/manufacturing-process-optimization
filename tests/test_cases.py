import numpy as np
from manufacturing_optimization import (
    case01_production_maintenance as c1,
    case02_milkrun as c2,
    case03_agv_assignment as c3,
    case04_worker_rotation as c4,
    case05_quality_mdp as c5,
    case06_cnc_process as c6,
    case07_tooling_scheduling as c7,
    case08_conwip as c8,
    case09_energy_scheduling as c9,
    case10_cutting_stock as c10,
)


def test_case01():
    result = c1.solve()
    assert len(result["assignment"]) == 6
    assert result["makespan"] > 0


def test_case02():
    result = c2.solve()
    assert result["service_level"] > 0.95
    assert np.all(result["inventory"] >= -1e-8)


def test_case03():
    result = c3.solve()
    assert np.all(result["battery_slack"] >= -1e-9)


def test_case04():
    result = c4.solve()
    assert result["schedule"].sum() == 12
    assert result["max_risk"] > 0


def test_case05():
    result = c5.solve()
    assert result["bellman_residual"] < 1e-7
    assert set(result["policy"]) == set(c5.STATES)


def test_case06():
    result = c6.solve()
    assert result["success"]
    assert result["roughness"] < 4.0


def test_case07():
    result = c7.solve()
    assert len(result["press_0"]) + len(result["press_1"]) == 6


def test_case08():
    result = c8.solve(caps=range(2, 6), seeds=range(3))
    assert 2 <= result["wip_cap"] <= 5
    assert result["throughput"] > 0


def test_case09():
    result = c9.solve()
    assert len(set(result["start_times"])) >= 2
    assert result["peak_power"] >= 4


def test_case10():
    result = c10.solve()
    assert np.all(result["produced"] >= result["demand"])
    assert result["total_waste"] >= 0
