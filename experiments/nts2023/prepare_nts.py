"""Step 2: convert NTS 2023 activity schedules to OASIS format and derive preferred times per segment.

Segments are employment group x household zone x day type (config.SEGMENTS). The preferred start time and duration of
an activity type in a segment are the medians of its observed episodes in the segment. If the segment has fewer than
config.MIN_SEGMENT_EPISODES episodes of the type, coarser groups are used in turn: employment x day type, employment,
whole population.

Outputs (in outputs/prepared/):
- persons.csv: pid, segment (and its components), n_activities and whether OASIS can represent the day
- schedules.csv: OASIS-format schedules (hours, dawn/dusk labels), with the segment of each person-day
- desired_times.json: preferred start time and duration per segment and activity type, with the group they come from
"""
import json
from typing import List

import numpy as np
import pandas as pd

from oasisabm.activity import Schedule
from oasisabm.helper_func import parse_df_schedule

import config as cfg


def load_activities() -> pd.DataFrame:
    acts = pd.read_csv(cfg.ACTIVITIES_PATH).sort_values(["pid", "seq"]).reset_index(drop=True)
    return acts


def load_segments() -> pd.DataFrame:
    attrs = pd.read_csv(cfg.ATTRIBUTES_PATH, dtype=str, usecols=["pid", "day", "employment", "hh_zone"])
    attrs["employment"] = attrs.employment.fillna("unknown")
    attrs["employment_group"] = attrs.employment.map(cfg.EMPLOYMENT_GROUPS)
    attrs["zone"] = attrs.hh_zone
    attrs["day_type"] = attrs.day.map(cfg.day_type)
    attrs["segment"] = [cfg.segment_of(d, e, z) for d, e, z in zip(attrs.day, attrs.employment, attrs.hh_zone)]
    return attrs[["pid", "segment", "employment_group", "zone", "day_type"]]


def clean(acts: pd.DataFrame) -> pd.DataFrame:
    """Drops zero-duration activities (except the first and last ones) and merges consecutive activities of the same type,
    which OASIS cannot represent."""
    acts = acts.copy()
    first = acts.pid != acts.pid.shift()
    last = acts.pid != acts.pid.shift(-1)
    acts = acts[(acts.end > acts.start) | first | last]

    run = ((acts.pid != acts.pid.shift()) | (acts.act != acts.act.shift())).cumsum()
    merged = acts.groupby(run).agg(pid=("pid", "first"), act=("act", "first"), start=("start", "min"), end=("end", "max"))
    merged["seq"] = merged.groupby("pid").cumcount() + 1
    return merged.reset_index(drop=True)


def representable(acts: pd.DataFrame) -> pd.Series:
    """True for days that start and end at home, leave home, and do not exceed the maximum number of episodes per type."""
    g = acts.groupby("pid")
    starts_home = g.act.first() == cfg.HOME
    ends_home = g.act.last() == cfg.HOME
    n = g.size()

    first = acts.pid != acts.pid.shift()
    last = acts.pid != acts.pid.shift(-1)
    middle = acts[~first & ~last]
    counts = middle.groupby(["pid", "act"]).size().unstack(fill_value=0).reindex(n.index, fill_value=0)
    caps = pd.Series({a: cfg.MAX_EPISODES.get(a, 0) for a in counts.columns})
    within_caps = (counts <= caps).all(axis=1)
    return starts_home & ends_home & (n >= 3) & within_caps


def to_oasis(acts: pd.DataFrame) -> pd.DataFrame:
    """Converts schedules to OASIS format: hours, dawn/dusk for the first/last home activity, labels '{type}_{k}'."""
    df = acts.copy()
    df["start_time"] = df.start / 60
    df["end_time"] = df.end / 60
    df["duration"] = df.end_time - df.start_time
    df["act_label"] = df.act
    df["act_id"] = df.act.map(cfg.ACT_ID)
    first = df.pid != df.pid.shift()
    last = df.pid != df.pid.shift(-1)
    k = df.groupby(["pid", "act"]).cumcount() + 1
    middle = ~first & ~last
    if cfg.HOME in df.act.values:
        # mid-day home episodes are numbered from 1, after dawn
        k = k.where(df.act != cfg.HOME, k - 1)
    df["label"] = df.act + "_" + k.astype(str)
    df.loc[first & (df.act == cfg.HOME), "label"] = "dawn"
    df.loc[last & (df.act == cfg.HOME), "label"] = "dusk"
    df.loc[~middle & (df.act != cfg.HOME), "label"] = df.act  # not representable anyway
    df["location"] = cfg.LOCATION
    df["mode"] = None
    return df[["pid", "act_id", "act_label", "label", "start_time", "end_time", "duration", "location", "mode"]]


def desired_times(oasis: pd.DataFrame, persons: pd.DataFrame) -> dict:
    """Preferred start time and duration per segment and activity type: medians of the observed episodes of the segment,
    or of the finest coarser group with enough episodes."""
    df = oasis.merge(persons, on="pid")
    df = df[~df.label.isin(["dawn", "dusk"]) & (df.act_label != cfg.HOME)]
    levels = [["employment_group", "zone", "day_type"], ["employment_group", "day_type"], ["employment_group"], []]
    medians = {
        tuple(level): df.groupby(level + ["act_label"]).agg(
            start=("start_time", "median"), duration=("duration", "median"), n=("pid", "size"))
        for level in levels
    }

    segments = persons.drop_duplicates("segment").set_index("segment")
    times = {}
    for segment in cfg.SEGMENTS:
        times[segment] = {}
        for act in cfg.OUT_OF_HOME_TYPES:
            for level in levels:
                table = medians[tuple(level)]
                key = tuple(segments.loc[segment, level]) + (act,) if level else act
                if key in table.index and (table.loc[key, "n"] >= cfg.MIN_SEGMENT_EPISODES or not level):
                    row = table.loc[key]
                    times[segment][act] = {
                        "desired_start_time": float(row.start),
                        "desired_duration": float(row.duration),
                        "n_episodes": int(row.n),
                        "group": "_".join(level) if level else "population",
                    }
                    break
    return times


def to_schedules(oasis: pd.DataFrame) -> List[Schedule]:
    """Converts OASIS-format schedules (one row per activity) into a list of Schedule objects, in order of pid."""
    schedules = []
    for _, df in oasis.groupby("pid", sort=True):
        sched = parse_df_schedule(df.reset_index(drop=True))
        for act in sched.list_act:
            act.mode = None
        schedules.append(sched)
    return schedules


def load_prepared() -> pd.DataFrame:
    return pd.read_csv(cfg.PREPARED_DIR / "schedules.csv")


def load_desired_times() -> dict:
    with open(cfg.PREPARED_DIR / "desired_times.json") as f:
        return json.load(f)


def main():
    cfg.PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_activities()
    acts = clean(raw)
    persons = load_segments()
    ok = representable(acts).rename("representable").reset_index()
    persons = persons.merge(ok, on="pid", how="left")
    persons["representable"] = persons.representable.fillna(False).astype(bool)
    persons = persons.merge(acts.groupby("pid").size().rename("n_activities").reset_index(), on="pid", how="left")

    oasis = to_oasis(acts)
    times = desired_times(oasis, persons)

    persons.to_csv(cfg.PREPARED_DIR / "persons.csv", index=False)
    with open(cfg.PREPARED_DIR / "desired_times.json", "w") as f:
        json.dump(times, f, indent=2)
    oasis = oasis.merge(persons[["pid", "segment", "representable"]], on="pid")
    oasis.to_csv(cfg.PREPARED_DIR / "schedules.csv", index=False)

    print(f"person-days: {len(persons)}, merged {len(raw) - len(acts)} activities")
    print(f"representable by OASIS: {persons.representable.mean():.1%}")
    pd.set_option("display.width", 250)
    print(persons.groupby("segment").agg(n=("pid", "size"), representable=("representable", "mean")).to_string())
    print(pd.DataFrame({s: {a: f"{v['desired_start_time']:.2f}h/{v['desired_duration']:.2f}h" for a, v in t.items()} for s, t in times.items()}).T.to_string())
    groups = pd.Series([v["group"] for t in times.values() for v in t.values()]).value_counts()
    print(f"preferred times by group used (segment x activity type):\n{groups.to_string()}")


if __name__ == "__main__":
    main()
