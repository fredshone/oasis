"""Step 4: generate synthetic schedules with the OASIS simulation model.

For each person-day, the model uses the population parameters and the preferred times of its segment, and is solved once
with freshly drawn error terms (seed = SEED + index of the person-day).

Outputs (in outputs/synthetic/{tag}/):
- synthetic_schedules.csv: pid, seq, act, start, end (minutes), as activities.csv
- synthetic_attributes.csv: binned attributes of the person-days
- solve_log.csv: pid, segment, solve status, runtime (seconds), whether the time limit was hit

Usage: python generate.py [--tag oasis] [--sample N] [--error-std 1.0] [--error-per-activity] [--pass pass2]
"""
import argparse
import json
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from oasisabm.error_terms import EVError, GaussianError
from oasisabm.input_data import activity_copies
from oasisabm.simulation import MIP

import config as cfg

SEED = 2024
TIME_LIMIT = 10  # seconds
CHUNK_SIZE = 200


def load_parameters(tag: str) -> Dict:
    """Parameters (the same for all segments) and preferred times per segment."""
    with open(cfg.PARAMS_DIR / f"{cfg.MODEL}_{tag}.json") as f:
        estimates = json.load(f)
    return {segment: {"parameters": estimates["parameters"], "desired_times": times}
            for segment, times in estimates["desired_times"].items()}


def build_activities(segment_params: Dict) -> List:
    return activity_copies(
        cfg.MAX_EPISODES, segment_params["parameters"], segment_params["desired_times"],
        act_id=cfg.ACT_ID, location=cfg.LOCATION, home=cfg.HOME,
    )


def utility_parameters(error_std: float) -> Dict:
    return {
        "error_w": GaussianError(std=error_std),
        "error_x": GaussianError(std=error_std),
        "error_d": GaussianError(std=error_std),
        "error_z": GaussianError(std=error_std),
        "error_ev": EVError(),
    }


def to_minutes(solution: pd.DataFrame) -> pd.DataFrame:
    """Converts a solution to activities.csv format: minutes, home for dawn/dusk, consecutive activities of the same
    type merged, and the day ending at 1440."""
    starts = np.round(solution.start_time.values * 60).astype(int)
    acts = solution.act_label.values
    starts[0] = 0
    ends = np.append(starts[1:], 24 * 60)
    df = pd.DataFrame({"act": acts, "start": starts, "end": ends})
    df = df[df.end > df.start]  # activities shorter than 30 seconds disappear after rounding
    run = (df.act != df.act.shift()).cumsum()
    df = df.groupby(run).agg(act=("act", "first"), start=("start", "min"), end=("end", "max")).reset_index(drop=True)
    df["seq"] = np.arange(1, len(df) + 1)
    return df[["seq", "act", "start", "end"]]


def solve_chunk(persons: pd.DataFrame, params: Dict, error_std: float, error_per_activity: bool) -> List:
    activities = {segment: build_activities(p) for segment, p in params.items()}
    results = []
    for pid, segment, seed in persons[["pid", "segment", "seed"]].itertuples(index=False):
        np.random.seed(seed)
        model = MIP(
            activities[segment], utility_parameters(error_std), travel_times=None, period=cfg.PERIOD,
            allow_repeats=False, min_duration=cfg.MIN_DURATION, require_out_of_home=True,
            error_per_activity=error_per_activity, sequence_error=cfg.SEQUENCE_ERROR, time_limit=TIME_LIMIT, threads=1,
        )
        start = time.time()
        model.initialize()
        solution, runtime, _ = model._solve()
        status = model.solve_details.status if model.solve_details else "none"
        log = {
            "pid": pid, "segment": segment, "status": status, "solved": solution is not None,
            "time_limit": "time limit" in status, "solve_time": runtime, "total_time": time.time() - start,
        }
        schedule = None
        if solution is not None:
            schedule = to_minutes(solution)
            schedule.insert(0, "pid", pid)
        results.append((log, schedule))
    return results


def generate(persons: pd.DataFrame, params: Dict, error_std: float = 1.0, error_per_activity: bool = False,
             n_jobs: int = -1, verbose: int = 0):
    """Generates one schedule per person-day. Returns schedules (minutes) and the solve log."""
    persons = persons.reset_index(drop=True).copy()
    persons["seed"] = SEED + persons.index
    chunks = [persons.iloc[i:i + CHUNK_SIZE] for i in range(0, len(persons), CHUNK_SIZE)]
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(solve_chunk)(chunk, params, error_std, error_per_activity) for chunk in chunks
    )
    results = [r for chunk in out for r in chunk]
    log = pd.DataFrame([r[0] for r in results])
    schedules = pd.concat([r[1] for r in results if r[1] is not None], ignore_index=True)
    return schedules, log


def load_persons(sample: Optional[int] = None, seed: int = SEED) -> pd.DataFrame:
    persons = pd.read_csv(cfg.PREPARED_DIR / "persons.csv")[["pid", "segment"]]
    if sample:
        persons = persons.sample(sample, random_state=seed)
    return persons.sort_values("pid").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="oasis")
    parser.add_argument("--pass", dest="params_tag", default="pass2")
    parser.add_argument("--sample", type=int, default=None, help="number of person-days (default: all)")
    parser.add_argument("--error-std", type=float, default=1.0)
    parser.add_argument("--error-per-activity", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    out_dir = cfg.SYNTHETIC_DIR / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    persons = load_persons(args.sample)
    params = load_parameters(args.params_tag)
    start = time.time()
    schedules, log = generate(persons, params, args.error_std, args.error_per_activity, args.n_jobs, verbose=5)
    elapsed = time.time() - start

    attributes = pd.read_csv(cfg.BINNED_ATTRIBUTES_PATH, dtype=str)
    attributes = attributes[attributes.pid.isin(schedules.pid.unique())]

    schedules.to_csv(out_dir / "synthetic_schedules.csv", index=False)
    attributes.to_csv(out_dir / "synthetic_attributes.csv", index=False)
    log.to_csv(out_dir / "solve_log.csv", index=False)
    settings = vars(args) | {"time_limit": TIME_LIMIT, "seed": SEED, "elapsed_seconds": elapsed}
    with open(out_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    print(f"{len(persons)} person-days in {elapsed:.0f}s")
    print(f"solved: {log.solved.mean():.2%}, hit time limit: {log.time_limit.mean():.2%}")
    print(f"solve time per schedule: mean {log.solve_time.mean():.2f}s, median {log.solve_time.median():.2f}s, max {log.solve_time.max():.1f}s")
    print(log.status.value_counts())


if __name__ == "__main__":
    main()
