import numpy as np
import pytest

from oasisabm.error_terms import GaussianError, EVError
from oasisabm.input_data import activity_copies
from oasisabm.simulation import MIP


ZERO = {"constant": 0, "early": 0, "late": 0, "short": 0, "long": 0}


def utility_params(std=1.0):
    return {
        "error_w": GaussianError(std=std),
        "error_x": GaussianError(std=std),
        "error_d": GaussianError(std=std),
        "error_z": GaussianError(std=std),
        "error_ev": EVError(),
    }


def solve(activities, std=1.0, seed=0, time_limit=30, **kwargs):
    np.random.seed(seed)
    model = MIP(activities, utility_params(std), travel_times=None, time_limit=time_limit, threads=1, **kwargs)
    model.initialize()
    solution, _, _ = model._solve()
    return model, solution


def check_valid(solution, period=24):
    assert solution.start_time.iloc[0] == pytest.approx(0, abs=1e-6)
    assert solution.end_time.iloc[-1] == pytest.approx(period, abs=1e-6)
    assert solution.duration.sum() == pytest.approx(period, abs=1e-6)
    # back to back: each activity starts when the previous one ends
    np.testing.assert_allclose(solution.start_time.values[1:], solution.end_time.values[:-1], atol=1e-6)
    assert solution.label.iloc[0] == "dawn"
    assert solution.label.iloc[-1] == "dusk"
    # consecutive activities never share a type
    assert (solution.act_label.values[1:] != solution.act_label.values[:-1]).all()


def test_zero_travel_gives_valid_24h_schedule():
    params = {
        "home": ZERO,
        "work": {"constant": 10, "early": -1, "late": -1, "short": -1, "long": -1},
        "shop": {"constant": 3, "early": -0.5, "late": -0.5, "short": -1, "long": -1},
    }
    desired = {
        "work": {"desired_start_time": 9, "desired_duration": 8},
        "shop": {"desired_start_time": 18, "desired_duration": 0.5},
    }
    activities = activity_copies({"home": 1, "work": 1, "shop": 1}, params, desired)
    model, solution = solve(activities, min_duration=1 / 60, allow_repeats=True)

    assert model.solve_status
    check_valid(solution)
    assert "work" in solution.act_label.values
    work = solution[solution.act_label == "work"].iloc[0]
    assert work.start_time == pytest.approx(9, abs=1)


def test_repeated_types_appear_when_favoured():
    params = {
        "home": ZERO,
        "shop": {"constant": 5, "early": 0, "late": 0, "short": -1, "long": -1},
    }
    desired = {"shop": {"desired_start_time": 12, "desired_duration": 1}}
    activities = activity_copies({"home": 2, "shop": 3}, params, desired)
    model, solution = solve(activities, std=0.01, min_duration=1 / 60, allow_repeats=True)

    assert model.solve_status
    check_valid(solution)
    shops = solution[solution.act_label == "shop"]
    assert len(shops) == 3
    # copies are scheduled in order
    assert list(shops.label) == ["shop_1", "shop_2", "shop_3"]
    # shops are separated by mid-day home episodes
    assert (solution.act_label == "home").sum() == 4


def test_without_repeats_each_group_appears_once():
    params = {"home": ZERO, "shop": {"constant": 5, "early": 0, "late": 0, "short": -1, "long": -1}}
    desired = {"shop": {"desired_start_time": 12, "desired_duration": 1}}
    activities = activity_copies({"home": 2, "shop": 3}, params, desired)
    _, solution = solve(activities, std=0.01, min_duration=1 / 60, allow_repeats=False)
    check_valid(solution)
    assert (solution.act_label == "shop").sum() == 1


def test_require_out_of_home():
    params = {"home": ZERO, "shop": {"constant": -50, "early": 0, "late": 0, "short": 0, "long": 0}}
    activities = activity_copies({"home": 1, "shop": 1}, params, {})
    _, solution = solve(activities, std=0.01, min_duration=1 / 60, allow_repeats=True)
    assert list(solution.label) == ["dawn", "dusk"]
    _, solution = solve(activities, std=0.01, min_duration=1 / 60, allow_repeats=True, require_out_of_home=True)
    check_valid(solution)
    assert "shop" in solution.act_label.values


def test_nts_sized_model_fits_community_edition():
    types = {"home": 2, "work": 2, "education": 1, "shop": 2, "other": 3, "escort": 3, "visit": 2, "medical": 1}
    params = {t: {"constant": -1, "early": -0.5, "late": -0.5, "short": -0.5, "long": -0.5} for t in types}
    params["home"] = ZERO
    desired = {t: {"desired_start_time": 10, "desired_duration": 1} for t in types if t != "home"}
    activities = activity_copies(types, params, desired)
    model, solution = solve(activities, time_limit=5, min_duration=1 / 60, allow_repeats=True, require_out_of_home=True)
    assert model.model.number_of_variables < 1000
    assert model.model.number_of_constraints < 1000
    assert model.solve_status
    check_valid(solution)
