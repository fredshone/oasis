"""Step 3a: build choice sets with Metropolis-Hastings sampling, for a sample of person-days.

The parameters are the same for all person-days, and the preferred times are those of the segment of each person-day.

The target weight of a schedule is exp(V), and alternatives get the sampling correction ln(k_j) - V(j).
First pass: V is a prior model, with a constant per episode equal to log(mean number of episodes per day) of each
out-of-home type, 0 for home (the reference, as in OASIS), and fixed penalties for deviations from the preferred times
(PRIOR_PENALTY per hour). A uniform target (as in the
original OASIS workflow) samples long random days that are perfectly separable from observed days, so the model
cannot be estimated (log-likelihood of 0).
Second pass (--target): V uses the first-pass estimates (the intended OASIS workflow).

Usage: python choice_sets.py [--n-persons 5000] [--target]
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from oasisabm.estimation import ChoiceSetGenerator

import config as cfg
from prepare_nts import load_prepared, load_desired_times, to_schedules

MH_PARAMS = {"n_iter": 3000, "n_burn": 1000, "n_skip": 10, "uniform": False, "log_weights": True, "rnd_term": 0}
N_ALT = 10
SEED = 2023
PRIOR_PENALTY = -0.5


def prior_parameters(oasis: pd.DataFrame) -> dict:
    """Prior utility parameters for the first pass: log of the mean number of episodes per day of each out-of-home type,
    0 for home, and fixed penalties."""
    n_days = oasis.pid.nunique()
    middle = oasis[~oasis.label.isin(["dawn", "dusk"])]
    counts = middle.act_label.value_counts()
    params = {cfg.HOME: {"constant": 0.0}}
    for t in cfg.OUT_OF_HOME_TYPES:
        params[t] = {"constant": float(np.log(max(counts.get(t, 0), 1) / n_days))}
        params[t].update({p: PRIOR_PENALTY for p in ["early", "late", "short", "long"]})
    return params


def with_desired_times(params: dict, desired_times: dict) -> dict:
    params = {t: dict(p) for t, p in params.items()}
    for t, times in desired_times.items():
        params[t]["desired_start_time"] = times["desired_start_time"]
        params[t]["desired_duration"] = times["desired_duration"]
    return params


def build(n_persons: int, target: bool, n_jobs: int) -> None:
    oasis = load_prepared()
    oasis = oasis[oasis.representable]
    pids = np.sort(oasis.pid.unique())
    rng = np.random.default_rng(SEED)
    if len(pids) > n_persons:
        pids = np.sort(rng.choice(pids, n_persons, replace=False))
    sample = oasis[oasis.pid.isin(pids)]

    if target:
        with open(cfg.PARAMS_DIR / f"{cfg.MODEL}_pass1.json") as f:
            base_params = json.load(f)["parameters"]
    else:
        base_params = prior_parameters(oasis)
    all_desired_times = load_desired_times()

    start = time.time()
    wides, longs, n_obs = [], [], 0
    for i, (segment, seg) in enumerate(sample.groupby("segment", sort=True)):
        seg_pids = np.sort(seg.pid.unique())
        desired_times = all_desired_times[segment]
        generator = ChoiceSetGenerator(
            to_schedules(seg), with_desired_times(base_params, desired_times), n_alt=N_ALT, mh_params=MH_PARAMS,
            activities=cfg.ACTIVITY_TYPES, desired_times=desired_times, max_episodes=cfg.MAX_EPISODES, min_activities=3,
            min_duration=cfg.MIN_SAMPLED_DURATION, circular=cfg.CIRCULAR, home=cfg.HOME, sample_in_walk=True,
            outfile=None, seed=SEED + 10_000 * i,
        )
        generator.run_parallel(n_cpus=n_jobs, verbose=0)
        df_wide, df_long, _ = generator.train_test_sets(train_ratio=1)
        df_wide.insert(0, "pid", seg_pids)
        df_long["pid"] = seg_pids[df_long.obs_id.values]
        df_long["obs_id"] += n_obs
        df_long["segment"] = segment
        n_obs += len(seg_pids)
        wides.append(df_wide)
        longs.append(df_long)
    df_wide = pd.concat(wides, ignore_index=True)
    df_long = pd.concat(longs, ignore_index=True)

    tag = "pass2" if target else "pass1"
    joblib.dump({"wide": df_wide, "long": df_long}, cfg.CHOICE_SET_DIR / f"{cfg.MODEL}_{tag}.joblib")
    n_av = df_wide[[f"av_{i}" for i in range(N_ALT)]].sum(axis=1)
    counts = [f'{t}:count' for t in cfg.ACTIVITY_TYPES]
    print(f"{tag}: {len(pids)} persons in {sample.segment.nunique()} segments in {time.time() - start:.0f}s, "
          f"mean alternatives per choice set {n_av.mean():.1f}, "
          f"mean activities chosen/sampled {df_long[df_long.choice == 1][counts].sum(axis=1).mean() + 2:.1f}"
          f"/{df_long[df_long.choice == 0][counts].sum(axis=1).mean() + 2:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-persons", type=int, default=5000)
    parser.add_argument("--target", action="store_true", help="use first-pass estimates as target weights (second pass)")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    cfg.CHOICE_SET_DIR.mkdir(parents=True, exist_ok=True)
    build(args.n_persons, args.target, args.n_jobs)


if __name__ == "__main__":
    main()
