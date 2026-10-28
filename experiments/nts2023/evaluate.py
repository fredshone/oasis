"""Step 5: score OASIS and caveat synthetic schedules against NTS 2023 with acteval.

Outputs (in outputs/evaluation/):
- summary.csv: domain-level distances (lower is better)
- groups.csv: group-level distances
- domains_by_attribute.csv: domain-level distances per attribute in SPLIT_ON
- acteval/: all acteval outputs (result.save)
- runtime.csv: OASIS solve statistics

Usage: python evaluate.py [--oasis oasis ...] [--no-caveat]
"""
import argparse

import pandas as pd
from acteval import compare

import config as cfg


def load_schedules(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["pid", "act", "start", "end"]]


def load_attributes(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df[["pid"] + cfg.SPLIT_ON]


def as_str_pid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pid"] = df.pid.astype(str)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oasis", nargs="*", default=["oasis"], help="tags of OASIS runs in outputs/synthetic")
    parser.add_argument("--no-caveat", action="store_true")
    parser.add_argument("--out", default=None, help="output directory (default: outputs/evaluation)")
    args = parser.parse_args()

    out_dir = cfg.EVAL_DIR if args.out is None else cfg.OUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    observed = as_str_pid(load_schedules(cfg.ACTIVITIES_PATH))
    observed_attributes = as_str_pid(load_attributes(cfg.BINNED_ATTRIBUTES_PATH))

    synthetic, synthetic_attributes, runtime = {}, {}, []
    for tag in args.oasis:
        run_dir = cfg.SYNTHETIC_DIR / tag
        synthetic[tag] = as_str_pid(load_schedules(run_dir / "synthetic_schedules.csv"))
        synthetic_attributes[tag] = as_str_pid(load_attributes(run_dir / "synthetic_attributes.csv"))
        log = pd.read_csv(run_dir / "solve_log.csv")
        runtime.append({
            "model": tag, "n_person_days": len(log), "solved": log.solved.mean(), "failed": 1 - log.solved.mean(),
            "hit_time_limit": log.time_limit.mean(), "mean_solve_seconds": log.solve_time.mean(),
            "median_solve_seconds": log.solve_time.median(), "mean_total_seconds": log.total_time.mean(),
        })

    if not args.no_caveat:
        for name, run_dir in cfg.CAVEAT_MODELS.items():
            synthetic[name] = as_str_pid(load_schedules(run_dir / "synthetic_schedules.csv"))
            synthetic_attributes[name] = as_str_pid(load_attributes(run_dir / "synthetic_attributes.csv"))

    result = compare(
        observed, synthetic, target_attributes=observed_attributes, synthetic_attributes=synthetic_attributes,
        split_on=cfg.SPLIT_ON,
    )

    summary = result.summary()
    summary.to_csv(out_dir / "summary.csv")
    result.at("groups").distances.to_csv(out_dir / "groups.csv")
    result.at("domains", "by_attribute").distances.to_csv(out_dir / "domains_by_attribute.csv")
    result.save(str(out_dir / "acteval"))
    pd.DataFrame(runtime).to_csv(out_dir / "runtime.csv", index=False)

    pd.set_option("display.width", 200)
    print(summary.round(3))
    print(result.at("groups").distances.round(3))
    print(pd.DataFrame(runtime).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
