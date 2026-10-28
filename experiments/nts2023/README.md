# OASIS on NTS 2023 activity schedules

This experiment fits OASIS to the NTS 2023 activity schedules, generates a synthetic schedule for every observed person-day, and scores the results with [acteval](https://github.com/fredshone/acteval) alongside four caveat models trained on the same data.

It records:
- how the OASIS code had to change for this use case and data;
- the three model versions we ran;
- what we learnt from them.

## Summary

- **OASIS needs real changes before it runs on NTS activity only data.** The main ones are:
  - a no-trip version of the scheduling MIP;
  - a working parameter-estimation pipeline (the repo has none);
  - fixes to the Metropolis-Hastings choice-set sampler;
  - activity types, preferred times and parameters passed in as inputs, instead of hard-coded globals.
- **The best version (v3) stays close to the published OASIS specification.** It uses:
  - one copy of each activity type;
  - a 1-hour minimum duration;
  - penalties pooled by flexibility;
  - one set of parameters for the whole population;
  - preferred start times and durations per segment (employment × household zone × day).
- **v3 is the best of our OASIS runs, but well behind the VAE-based caveat models on distribution fit.**
  - Participations, timing and transitions are 2–6× worse than the VAE models.
  - It beats the conditional caveat model on participations and timing.
  - Its schedules are much more varied (creativity 0.46 against about 0.005 for the VAE models). OASIS rarely copies an observed schedule.
- **The number of activities per day is controlled mostly by the error scale and by how many activity slots the MIP has.** Removing trips matters much less.

## Data

`~/Data/foundata/out/nts/2023/`:
- `activities.csv`: `pid,seq,act,zone,start,end` in minutes.
- `attributes.csv` and `attributes_binned.csv`: one row per `pid`.

There are 56,734 person-days. In this data:
- Activities run back to back from 0 to 1440 minutes, with no trips. Travel time is therefore included in activity durations.
- `zone` is always `unknown`.
- Every day starts and ends at home, and no day is spent entirely at home.
- There are eight activity types: `home, work, education, shop, other, escort, visit, medical`.
- Days have 3–17 activities (median 3). About 20% of days repeat a non-home activity type.

Training and evaluation both use the full dataset, as in the caveat comparison.

## How OASIS had to change

### Changes to the library (`oasisabm/`)

| Change | Files | Why |
|---|---|---|
| Imports use the package name (`from oasisabm.x import ...`), `oasisabm/__init__.py` added, and the example runners use paths relative to the repo | all modules, `runner*.py` | Before this, the code only ran from inside `oasisabm/`. |
| **No-trip MIP** (`travel_times=None` → `MIP._initialize_no_travel`). There are no travel time, cost or mode variables. Durations satisfy `sum(d) == period`, and activities follow each other with no gap. | `simulation.py` | NTS schedules have no trips or locations. The original behaviour is kept when `travel_times` is given. |
| The no-trip MIP is **compact**: sequence arcs only between feasible pairs (never into dawn, out of dusk, or between two activities of the same type), and binary × continuous products are linearised (`_product_with_w`). Penalties must be ≤ 0. | `simulation.py` | The CPLEX Community Edition is limited to 1000 variables and 1000 constraints. |
| New MIP options: `allow_repeats`, `min_duration`, `require_out_of_home`, `error_per_activity`, `sequence_error`, `threads` | `simulation.py` | Each departure from OASIS can be switched on or off, so we could test which ones matter (see the ablation). |
| `sequence_error='all'` reproduces the original sequence error term w[a]·Σ_b error_z(z[a,b]). With one activity's successors fixed, this equals w[a]·(e_z1 + (n−1)·e_z0), which is linear in w. `'used'` keeps only the error of the transition actually taken. | `simulation.py` | Our first version (`'used'`) added fewer error terms than OASIS does. |
| `activity_copies()` builds `dawn`, `dusk` and K copies per activity type (`shop_1`, `shop_2`, …), each with its own preferred times and parameters | `input_data.py` | Activity types, preferred times and parameters become inputs instead of globals. Default act IDs no longer crash on unknown types. |
| `Activity` and `Schedule.compute_utility` accept preferred times and per-type parameters. A random term of 0 is no longer treated as "draw one" (`if rnd_term is None`). Missing `travel_time` parameters default to 0. | `activity.py` | Needed to use NTS activity types and to give the sampler a deterministic target. |
| **Metropolis-Hastings sampler fixes:** `random_walk` can use log-weights (target exp(V)); can yield rejected states, so samples follow the target distribution; rejects invalid proposals (not starting/ending at home, too many episodes, too short, home-only); and no longer hard-codes `business_trip:constant = 0`. | `metropolis_hastings.py` | The original sampler used V itself as the weight (meaningless with negative utilities) and only kept accepted states. Without these fixes, the choice sets did not follow the target distribution and the sampling correction was wrong. |
| Operator bugs: `Assign` now splits the activity it is inserted into. `InflateDeflate` looks up the activity, not its label. `Swap` checks the neighbour exists and swaps labels correctly. `AddAnchor` is capped at 23 anchors. `MetaOperator` passes `kwargs` through. `Schedule.which_activity` no longer returns the next activity instead of the current one. | `operators.py`, `activity.py` | Each of these silently produced wrong schedules or raised errors that the original `except: continue` hid. |
| **Estimation pipeline:** `ChoiceSetGenerator` takes seeds, preferred times, episode limits, minimum durations and an option for circular start-time deviations. `schedule_features` computes Biogeme variables per type. `train_test_sets` was rewritten. The sampling correction is ln k_j − ln q(j) (Flötteröd & Bierlaire 2013). | `estimation.py` | The repo only produced choice sets, and `train_test_sets` had bugs: the `choice` lambda used the outer loop's `i`, mislabelling the chosen alternative, and `test_ids = other_ids.extend(...)` was always `None`. The original correction also did not match the MH target. |
| Tests | `tests/test_no_travel.py`, `tests/test_estimation.py` | They check that the no-trip MIP gives valid 24h schedules, that repeats work, and that the estimation features and correction are right. |

The repo contains **no Biogeme model**. We wrote one in `estimate.py` (see below).

### What the experiment scripts add (`experiments/nts2023/`)

| Script | What it does |
|---|---|
| `config.py` | Activity types, episode limits, flexibility classes, segments, paths, caveat models and evaluation splits |
| `prepare_nts.py` | Converts minutes to hours and the first and last home episodes to `dawn` and `dusk`. Marks which days the model can represent. Computes preferred times as segment medians, with fallback to coarser groups. |
| `choice_sets.py` | MH choice sets for 5,000 sampled days, using each day's segment preferred times. Pass 1 uses a prior target (home 0, other constants = log mean episodes, penalties −0.5). Pass 2 uses the pass-1 estimates as the target. Settings: 3,000 iterations, 1,000 burn-in, every 10th state, 10 alternatives. |
| `estimate.py` | Biogeme 3.3.5 multinomial logit with the sampling correction as an offset. Writes parameters in OASIS format. |
| `tune.py` | Scores error scales on 2,000 days with acteval and picks the one with the lowest sum of density scores. |
| `generate.py` | One MIP per observed person-day (joblib, a seed per day, 10s CPLEX time limit, 1 thread each), using the day's segment preferred times |
| `evaluate.py` | Scores the OASIS runs and the caveat models with `acteval.compare` in the same run, split by person attributes |

Run them in this order:

```bash
python prepare_nts.py
python choice_sets.py                     # pass 1
python estimate.py --pass pass1
python choice_sets.py --target            # pass 2
python estimate.py --pass pass2
python tune.py
python generate.py --tag oasis --pass pass2 --error-std 0.25
python evaluate.py --oasis oasis oasis_v2 oasis_v1
```

## The model (v3, current)

**How it matches OASIS:**
- One copy of each activity type, as OASIS allows at most one activity per group.
  - Days that repeat an activity type, or return home more than once, can't be represented. They are left out of estimation; 84.1% of days are representable.
  - Generation still produces a schedule for every person-day.
- 1-hour minimum duration in the simulation.
- The sequence error is summed over all activities (`'all'`). Error terms are Gaussian and shared by all activities, as in OASIS.
- Start-time deviations wrap around midnight in estimation (as in OASIS) and are linear in the simulation (as in OASIS).
- One set of parameters for everyone, with no attribute terms:
  - an alternative-specific constant per activity type, with home as the reference (0);
  - early, late, short and long penalties pooled by flexibility class, as in OASIS:
    - not flexible (NF): work, education, escort, medical;
    - flexible (F): shop, other, visit.

**Preferred times per segment.** OASIS reads preferred times as inputs and does not say how they were obtained. We use medians per segment, defined as employment (ft, pt, child/student, retired, not working) × household zone (urban, suburban, rural) × day type (weekday, Saturday, Sunday). That gives 45 segments.
- When a segment has fewer than 30 episodes of a type, the median comes from the next coarser group: employment × day type, then employment, then the whole population.
- Of the 315 activity–segment cells, 220 use the full segment. 32 fall back to employment × day type, 54 to employment, and 9 to the whole population.
- The same preferred times are used in estimation and simulation. This is the only way the model depends on person attributes.

**Pass-2 estimates** (n = 5,000; LL −11,110 against −15,023 for constants only; no penalty at its bound):

| | early | late | short | long |
|---|---|---|---|---|
| NF | −0.56 | −0.56 | −0.69 | −0.78 |
| F | −0.38 | −0.34 | −0.25 | −0.61 |

| work | education | shop | other | escort | visit | medical |
|---|---|---|---|---|---|---|
| −1.61 | −1.39 | −1.88 | −1.49 | −1.62 | −3.07 | −3.11 |

**Error scale.** OASIS's default standard deviation of 1 gives far too many activities (5.4 a day against 3.9 observed). A standard deviation of 0.25 matches the observed count:

| error std | sum of density scores | creativity | activities per day (observed 3.92) | hit 10s limit |
|---|---|---|---|---|
| **0.25** | **0.320** | 0.396 | 3.92 | 7% |
| 0.1 | 0.562 | 0.492 | 3.21 | 0% |
| 0.5 | 0.568 | 0.311 | 4.73 | 22% |
| 1 (OASIS default) | 0.888 | 0.263 | 5.44 | 32% |

**Runtime.** All 56,734 person-days solved in 92 minutes on all cores.
- Solve time per day: mean 1.49s, median 0.12s.
- 8.7% hit the 10s time limit. Their schedules are the best found in time, not proven optimal.

### Remaining departures from OASIS

- **Trips and locations:** none, because the data has none. Travel time is absorbed into activity durations and constants.
- **Home-only days** are not allowed (`require_out_of_home`), because none are observed.
- **Same-type sequences** (e.g. shop → shop) are not allowed.
- **Penalties** are bounded at ≤ 0 in estimation, which the linearised MIP requires. No estimate reached the bound.
- **Sampled alternatives** may have activities as short as 1 minute, so the observed short episodes have comparable alternatives. The simulation keeps the 1-hour minimum.
- **Constants:** all are estimated. OASIS fixes the escort, errands and business-trip constants at 0.
- **Sampler:** uses the fixed Metropolis-Hastings sampler and its sampling correction (see above).
- **Error scale** is 0.25, chosen by tuning, rather than 1.

## Experiments

### v1: copies, separate segments (`outputs/archive_v1_segments_copies/`)

This was the first attempt to reproduce repeated activities.
- **Copies:** up to K copies per type (home 2, work 2, shop 2, other 3, escort 3, visit 2, education 1, medical 1), ordered by start time.
- **Durations:** 1-minute minimum.
- **Sequence error:** only on the transition used.
- **Penalties:** per activity type.
- **Parameters and preferred times:** a separate set per segment, for 8 segments (weekday/weekend × ft, pt, child/student, not working).
- **Errors:** drawn per activity type, with tuned std 0.1.

The schedules had too many short activities. Homogeneity was 98.6%, so almost every schedule was a variation on the same pattern.

### Ablation: what drives the number of activities?

The test used 400 days with v1 parameters and error std 1:

| variant | activities per day | out-of-home per day | share under 30 min | median duration (min) |
|---|---|---|---|---|
| observed | 3.94 | 1.62 | 0.16 | 100 |
| v1 (copies, 1 min, used z) | 6.16 | 3.02 | 0.57 | 19 |
| 1h minimum | 5.21 | 2.38 | 0.00 | 60 |
| one copy per type | 4.26 | 1.75 | 0.46 | 60 |
| original sequence error | 8.68 | 5.80 | 0.58 | 1 |
| OASIS-like (one copy, 1h, all z) | 5.19 | 2.76 | 0.00 | 60 |
| v1, std 0.01 | 3.38 | 1.19 | 0.21 | 61 |
| OASIS-like, std 0.01 | 3.00 | 1.00 | 0.00 | 390 |

Conclusions:
- Extra activities come from **error terms**. Each scheduled activity collects positive error draws, so with large errors more activities are always better.
- More **slots** (copies) and **cheap short activities** (1-minute minimum) make this worse.
- The original sequence error adds (n−1)·e_z0 per activity, which makes it worse still.
- Removing **trips** is not the main cause. The estimated constants already absorb the per-episode cost that travel would add.
- With almost no error, the model under-schedules (3.0 activities per day). The error scale is therefore the main way to control activity counts.

### v2: OASIS-like, population preferred times (`outputs/archive_v2_population_times/`)

- One copy of each type, 1-hour minimum and `sequence_error='all'`.
- Penalties pooled by flexibility, with one set of parameters.
- **Preferred times:** medians over the whole population.
- **Error:** tuned std 0.25 gave 3.80 activities per day.
- **Runtime:** 10.8% hit the time limit.
- **Pass-2 penalties:** NF −0.56/−0.57/−0.62/−0.80 and F −0.36/−0.32/−0.19/−0.60 (early/late/short/long). The work constant was poorly identified (t = −1.0).

### v3: preferred times per segment (current, `outputs/`)

This is v2 with preferred times per employment × zone × day-type segment. The penalties hardly change, but the constants are now well identified (work t = −5.8, education t = −3.0).

## Results

All models were scored in **one acteval 0.3.0 run** (lower is better). Scores from earlier acteval versions or other runs are not comparable.

| domain | **v3** | v2 | v1 | caveat vae | caveat cvae | caveat conditional | caveat cpvae |
|---|---|---|---|---|---|---|---|
| participations | **0.184** | 0.208 | 0.228 | 0.095 | 0.094 | 0.409 | 0.047 |
| timing | **0.118** | 0.151 | 0.151 | 0.029 | 0.032 | 0.222 | 0.020 |
| transitions | **0.018** | 0.022 | 0.024 | 0.006 | 0.005 | 0.015 | 0.003 |
| creativity | 0.460 | 0.488 | 0.515 | 0.005 | 0.004 | 0.249 | 0.005 |
| feasibility | 0 | 0 | 0 | 0.003 | 0.017 | 0 | 0.016 |

Scores by the attributes used for segmentation:

| | v3 | v2 | v1 |
|---|---|---|---|
| participations by employment | 0.202 | 0.218 | 0.340 |
| participations by hh_zone | 0.178 | 0.200 | 0.199 |
| participations by day | 0.172 | 0.200 | 0.212 |
| timing by employment | 0.121 | 0.156 | 0.199 |
| timing by hh_zone | 0.113 | 0.147 | 0.138 |
| timing by day | 0.115 | 0.147 | 0.145 |

Results are in `outputs/evaluation/`:
- `summary.csv`
- `groups.csv`
- `domains_by_attribute.csv`
- `runtime.csv`
- acteval reports

Synthetic schedules are in `outputs/synthetic/`.

## Conclusions

1. **Out of the box, OASIS can't be fitted to diary data like the NTS.**
   - It assumes trips and locations.
   - It hard-codes activity types and preferred times.
   - It has no estimation code.
   - Its MH sampler does not sample from its stated target.
   - With trips removed, the MIP has to be reformulated and kept compact to fit within the free CPLEX limits.
2. **The published structure (one episode per type, 1-hour minimum) fits NTS days better than adding copies.**
   - It can't represent the 16% of days with repeated activities.
   - Even so, adding copies made the fit worse: the extra slots fill up with short activities.
3. **The error scale is effectively a free parameter that sets how many activities people do.**
   - It is not estimated, and the OASIS default (std 1) is far too large here.
   - We tuned it to the observed distribution, which a fair comparison should note.
4. **Preferred times are the only route for person attributes in OASIS.**
   - Making them depend on employment, zone and day type gave the largest single improvement: participations −12% and timing −22% against v2.
   - It improved fit within every attribute group, including ones not used for segmenting.
5. **Against the deep generative models, OASIS is much worse on distribution fit but is always feasible and highly varied.**
   - The gap is largest for timing (about 4–6×).
   - That suggests single preferred times with linear penalties are too rigid to reproduce the spread of observed start times and durations.

## Caveats

- `access_egress_distance` is derived from each person's own diary trips (foundata `nts.py`). A left join makes it close to the person's mean trip distance, so it leaks information about the schedule. foundata's own anomaly checks exclude it, and it should be removed from `SPLIT_ON`.
- Choice sets and estimation use a sample of 5,000 days. One segment had no sampled days, but it still has preferred times for simulation.
- 8.7% of generated days stopped at the solver time limit and may not be optimal.
