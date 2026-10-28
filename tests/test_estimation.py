import numpy as np
import pandas as pd
import pytest

from oasisabm.activity import Activity, Schedule
from oasisabm.estimation import ChoiceSetGenerator
from oasisabm.metropolis_hastings import is_valid
from oasisabm.operators import Swap, Assign

TYPES = ["home", "work", "shop"]
DESIRED = {"work": {"desired_start_time": 9, "desired_duration": 8}, "shop": {"desired_start_time": 18, "desired_duration": 1}}


def schedule(*acts):
    return Schedule(list_act=[Activity(label, start, duration, desired_times={}) for label, start, duration in acts])


def test_swap_exchanges_labels():
    np.random.seed(0)
    sched = schedule(("home", 0, 8), ("work", 8, 8), ("shop", 16, 1), ("home", 17, 7))
    sched.anchor_nodes = [8]
    new = Swap(1).apply_change(sched)
    # work swaps with its next or previous neighbour
    assert new.all_labels in (["home", "shop", "work", "home"], ["work", "home", "shop", "home"])


def test_assign_inside_activity_splits_it():
    np.random.seed(1)
    sched = schedule(("home", 0, 8), ("work", 8, 8), ("home", 16, 8))
    sched.anchor_nodes = [12.0]
    new = Assign(1, list_act=["shop"]).apply_change(sched)
    new.streamline()
    assert new.all_labels == ["home", "work", "shop", "work", "home"]
    assert sum(a.duration for a in new.list_act) == pytest.approx(24)


def test_is_valid():
    caps = {"home": 1, "work": 1, "shop": 2}
    assert is_valid(schedule(("home", 0, 8), ("work", 8, 8), ("home", 16, 8)), caps, 3)
    assert not is_valid(schedule(("home", 0, 8), ("work", 8, 16)), caps, 2)
    assert not is_valid(schedule(("home", 0, 24)), caps, 3)
    assert not is_valid(schedule(("home", 0, 8), ("work", 8, 4), ("home", 12, 1), ("work", 13, 3), ("home", 16, 8)), caps, 3)
    assert not is_valid(schedule(("home", 0, 8), ("shop", 8, 0.001), ("home", 8.001, 15.999)), caps, 3, min_duration=1 / 60)


def test_choice_sets_and_estimation_data():
    observed = [schedule(("home", 0, 8), ("work", 8, 9), ("home", 17, 7)),
                schedule(("home", 0, 17), ("shop", 17, 1), ("home", 18, 6))]
    generator = ChoiceSetGenerator(
        observed, None, n_alt=5, mh_params={"n_iter": 300, "n_burn": 50, "n_skip": 5, "uniform": True},
        activities=TYPES, desired_times=DESIRED, max_episodes={"home": 1, "work": 1, "shop": 2},
        min_activities=3, circular=False, outfile=None, sample_in_walk=True,
    )
    generator.run()
    wide, long, test = generator.train_test_sets(train_ratio=1)

    assert test == []
    assert len(wide) == 2
    assert (wide.choice == 0).all()
    assert (wide.av_0 == 1).all()
    chosen = long[long.choice == 1]
    assert list(chosen.alt_id) == [0, 0]
    assert chosen["work:count"].tolist() == [1, 0]
    assert chosen["work:short"].iloc[0] == 0 and chosen["work:long"].iloc[0] == pytest.approx(1)
    assert chosen["work:early"].iloc[0] == pytest.approx(1)
    # sampling correction: log of the number of times each alternative is in the choice set (uniform target)
    for obs, df in long.groupby("obs_id"):
        assert np.exp(df.prob_corr).round().sum() == 5
