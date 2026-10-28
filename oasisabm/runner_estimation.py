import joblib
from pathlib import Path

from oasisabm.estimation import ChoiceSetGenerator
from oasisabm.helper_func import parse_df_schedule

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA = joblib.load(DATA_DIR / 'example_data_estimation.joblib')
TT = joblib.load(DATA_DIR / 'example_tt.joblib')
PARAMS = DATA_DIR / 'target_params_032023.joblib'

MH_PARAMS = {"n_iter":1000,
"n_burn": 50,
 "n_skip": 1,
 "uniform": False,
}

N_ALT = 5

def main():
    data = parse_df_schedule(DATA)
    estimator = ChoiceSetGenerator(DATA, PARAMS, n_alt = 5, mh_params=MH_PARAMS)
    estimator.run()

    train_wide, train_long, test = estimator.train_test_sets()

if __name__ == '__main__':
    main()
