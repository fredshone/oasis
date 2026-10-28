"""Step 4 (tuning): rough tuning of the scale of the simulation error terms, which are not estimated.

Generates schedules for a sample of person-days with several error standard deviations, with error terms shared by all
activities (as in OASIS) or, with --per-activity, also drawn per activity type, and compares them with the observed schedules of the same
person-days with acteval (domain distances, lower is better).

Usage: python tune.py [--sample 2000] [--stds 0.1 0.25 0.5 1] [--pass pass2] [--per-activity]
"""
import argparse

import pandas as pd
from acteval import compare

import config as cfg
from generate import generate, load_parameters, load_persons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--stds", nargs="*", type=float, default=[0.1, 0.25, 0.5, 1])
    parser.add_argument("--pass", dest="params_tag", default="pass2")
    parser.add_argument("--per-activity", action="store_true", help="also try error terms drawn per activity type")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    persons = load_persons(args.sample, seed=1)
    params = load_parameters(args.params_tag)
    observed = pd.read_csv(cfg.ACTIVITIES_PATH)
    observed = observed[observed.pid.isin(persons.pid)][["pid", "act", "start", "end"]]

    synthetic, logs = {}, []
    for per_activity in [False, True] if args.per_activity else [False]:
        for std in args.stds:
            name = f"std{std:g}_{'per_activity' if per_activity else 'shared'}"
            schedules, log = generate(persons, params, std, per_activity, args.n_jobs)
            synthetic[name] = schedules[["pid", "act", "start", "end"]]
            n_acts = schedules.groupby("pid").size()
            logs.append({"run": name, "solved": log.solved.mean(), "hit_time_limit": log.time_limit.mean(),
                         "mean_solve_seconds": log.solve_time.mean(), "mean_activities": n_acts.mean()})
            print(logs[-1])

    result = compare(observed, synthetic)
    summary = result.summary().T
    summary["sum_density"] = summary[["participations", "transitions", "timing"]].sum(axis=1)
    summary = summary.join(pd.DataFrame(logs).set_index("run"))
    summary.loc["observed", "mean_activities"] = observed.groupby("pid").size().mean()

    out_dir = cfg.OUT_DIR / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"tuning_{args.params_tag}.csv")
    pd.set_option("display.width", 200)
    print(summary.round(3).sort_values("sum_density"))


if __name__ == "__main__":
    main()
