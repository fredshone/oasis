"""Step 3b: estimate a sampling-corrected multinomial logit model with Biogeme, for the whole population.

The utility of a schedule is the sum over its out-of-home episodes of
constant + early * early_deviation + late * late_deviation + short * short_deviation + long * long_deviation,
where deviations are from the preferred times of the segment of the person-day.
As in OASIS, constants are estimated per activity type, home is the reference (utility 0), and penalties are estimated
per flexibility class (F or NF, see config.FLEXIBILITY). Penalties are bounded above by 0, because the simulation model
(linearised) requires them to be negative or zero; estimates at the bound are reported.

Outputs outputs/parameters/all_{pass}.json, with the parameters in OASIS format, the preferred times per segment and
estimation statistics.

Usage: python estimate.py [--pass pass1]
"""
import argparse
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd

import biogeme.biogeme as bio
from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters

import config as cfg
from prepare_nts import load_desired_times

PENALTIES = ["early", "late", "short", "long"]


def column(name: str) -> str:
    """Biogeme-friendly column name."""
    return name.replace(":", "__")


def utilities(n_alt: int, constants_only: bool = False):
    constants = {t: Beta(f"{t}_constant", 0, None, None, 0) for t in cfg.OUT_OF_HOME_TYPES}
    # penalties are bounded at 0: the simulation model requires penalties <= 0
    penalties = {} if constants_only else {
        (fd, p): Beta(f"{fd}_{p}", -0.1, None, 0, 0) for fd in sorted(set(cfg.FLEXIBILITY.values())) for p in PENALTIES
    }

    util, av = {}, {}
    for j in range(n_alt):
        v = Variable(f"prob_corr_{j}")
        for t in cfg.OUT_OF_HOME_TYPES:
            v = v + constants[t] * Variable(column(f"{t}:count_{j}"))
            for p in PENALTIES:
                if (cfg.FLEXIBILITY[t], p) in penalties:
                    v = v + penalties[cfg.FLEXIBILITY[t], p] * Variable(column(f"{t}:{p}_{j}"))
        util[j] = v
        av[j] = Variable(f"av_{j}")
    return util, av


def estimate(df: pd.DataFrame, n_alt: int, name: str, constants_only: bool = False):
    df = df.rename(columns=column).drop(columns=["pid"]).astype(float)
    database = Database(name, df)
    util, av = utilities(n_alt, constants_only)
    logprob = loglogit(util, av, Variable("choice"))
    # run in a temporary directory: Biogeme writes its configuration and output files to the working directory
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            model = bio.BIOGEME(database, logprob, generate_html=False, generate_yaml=False, save_iterations=False)
            model.model_name = name
            results = model.estimate()
        finally:
            os.chdir(cwd)
    return results


def to_oasis_parameters(betas: dict) -> dict:
    """Parameters per activity type (OASIS simulation format). Home is the reference: all its parameters are 0."""
    params = {cfg.HOME: {"constant": 0.0} | {p: 0.0 for p in PENALTIES}}
    for t in cfg.OUT_OF_HOME_TYPES:
        params[t] = {"constant": betas.get(f"{t}_constant", 0.0)}
        for p in PENALTIES:
            params[t][p] = betas.get(f"{cfg.FLEXIBILITY[t]}_{p}", 0.0)
    return params


def run(model: str, tag: str) -> dict:
    data = joblib.load(cfg.CHOICE_SET_DIR / f"{model}_{tag}.joblib")
    df = data["wide"]
    n_alt = len([c for c in df.columns if c.startswith("av_")])

    results = estimate(df, n_alt, f"{model}_{tag}")
    null = estimate(df, n_alt, f"{model}_{tag}_constants", constants_only=True)

    betas = results.get_beta_values()
    table = pd.concat(get_pandas_estimated_parameters(results).values())
    ll, ll_constants = results.final_loglikelihood, null.final_loglikelihood
    at_bound = [n for n, v in betas.items() if any(n.endswith(p) for p in PENALTIES) and v > -1e-6]

    out = {
        "model": model,
        "pass": tag,
        "n_observations": len(df),
        "log_likelihood": ll,
        "log_likelihood_constants_only": ll_constants,
        "likelihood_ratio_test": 2 * (ll - ll_constants),
        "penalties_at_zero_bound": at_bound,
        "parameters": to_oasis_parameters(betas),
        "desired_times": load_desired_times(),
        "converged": bool(results.algorithm_has_converged),
        "estimates": table.to_dict(orient="records"),
    }
    with open(cfg.PARAMS_DIR / f"{model}_{tag}.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    print(f"{model} ({tag}): n={len(df)}, LL={ll:.1f}, LL constants only={ll_constants:.1f}, penalties at 0 bound: {at_bound}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="tag", default="pass1")
    args = parser.parse_args()

    cfg.PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = run(cfg.MODEL, args.tag)
    print(pd.DataFrame(out["estimates"]).round(3).to_string())


if __name__ == "__main__":
    main()
