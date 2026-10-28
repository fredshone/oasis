"""Shared settings for the NTS 2023 experiment."""
from pathlib import Path

DATA_DIR = Path("~/Data/foundata/out/nts/2023").expanduser()
ACTIVITIES_PATH = DATA_DIR / "activities.csv"
ATTRIBUTES_PATH = DATA_DIR / "attributes.csv"
BINNED_ATTRIBUTES_PATH = DATA_DIR / "attributes_binned.csv"

OUT_DIR = Path(__file__).resolve().parent / "outputs"
PREPARED_DIR = OUT_DIR / "prepared"
CHOICE_SET_DIR = OUT_DIR / "choice_sets"
PARAMS_DIR = OUT_DIR / "parameters"
SYNTHETIC_DIR = OUT_DIR / "synthetic"
EVAL_DIR = OUT_DIR / "evaluation"

HOME = "home"
LOCATION = "none"

# Model specification, as close as possible to OASIS as published:
# - one copy of each activity type (OASIS allows one activity per group), so days with repeated activities are not
#   used for estimation. For home, this is one mid-day return home.
# - minimum duration of 1 hour in the simulation (OASIS: w <= d, in hours)
# - sequence error terms summed over all activities (OASIS: sum_b error_z(z[a,b])), shared by all activities
# - one set of parameters for the whole population (no attribute terms), penalties pooled by flexibility
# - preferred times per segment (employment x zone x day type), used in both estimation and simulation
MAX_EPISODES = {
    "home": 1,
    "work": 1,
    "education": 1,
    "shop": 1,
    "other": 1,
    "escort": 1,
    "visit": 1,
    "medical": 1,
}
ACTIVITY_TYPES = list(MAX_EPISODES)
OUT_OF_HOME_TYPES = [t for t in ACTIVITY_TYPES if t != HOME]
ACT_ID = {t: i for i, t in enumerate(ACTIVITY_TYPES, 1)}

# flexible (F) or not flexible (NF) activity types: penalties are estimated per flexibility class, as in OASIS
FLEXIBILITY = {
    "work": "NF",
    "education": "NF",
    "escort": "NF",
    "medical": "NF",
    "shop": "F",
    "other": "F",
    "visit": "F",
}

PERIOD = 24  # hours
MIN_DURATION = 1  # hours, minimum duration of a scheduled activity in the simulation
MIN_SAMPLED_DURATION = 1 / 60  # hours, minimum duration of an activity in sampled alternatives (excludes zero-length activities)
SEQUENCE_ERROR = "all"
CIRCULAR = True  # deviations from preferred start times wrap around midnight in the estimation, as in OASIS

MODEL = "all"  # one set of parameters for all segments

# Segments for the preferred times: employment group x household zone x day type
EMPLOYMENT_GROUPS = {
    "ft-employed": "ft",
    "pt-employed": "pt",
    "student": "child_student",
    "unknown": "child_student",  # mostly children under 16
    "retired": "retired",
    "unemployed": "not_working",
    "other": "not_working",
}
ZONES = ["urban", "suburban", "rural"]
DAY_TYPES = ["weekday", "saturday", "sunday"]
SEGMENTS = [f"{e}_{z}_{d}" for e in sorted(set(EMPLOYMENT_GROUPS.values())) for z in ZONES for d in DAY_TYPES]
# minimum number of episodes of an activity type to compute its preferred times in a segment; otherwise, coarser groups
# are used in turn: employment x day type, employment, whole population
MIN_SEGMENT_EPISODES = 30

# caveat models trained on the same data, compared with OASIS
CAVEAT_LOGS = Path("~/Projects/caveat/logs/actvae").expanduser()
CAVEAT_MODELS = {
    "caveat_vae": CAVEAT_LOGS / "vae/vae_nrun0/version_0",
    "caveat_cvae": CAVEAT_LOGS / "cvae/cvae_nrun0/version_0",
    "caveat_conditional": CAVEAT_LOGS / "conditional/conditional_nrun0/version_0",
    "caveat_cpvae": CAVEAT_LOGS / "cpvae/cpvae-add_nrun0/version_0",
}
SPLIT_ON = ["age", "sex", "employment", "hh_income", "hh_zone", "day", "vehicles", "access_egress_distance"]


def day_type(day: str) -> str:
    return day if day in ("saturday", "sunday") else "weekday"


def segment_of(day: str, employment: str, zone: str) -> str:
    return f"{EMPLOYMENT_GROUPS[employment]}_{zone}_{day_type(day)}"
