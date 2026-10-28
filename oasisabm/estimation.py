
import joblib
import time
import warnings
import math
from copy import deepcopy

import numpy as np
import pandas as pd
from oasisabm import metropolis_hastings as mh

from datetime import datetime
from joblib import Parallel, delayed, cpu_count

from oasisabm.metropolis_hastings import random_walk
from oasisabm.helper_func import schedule_to_pandas
from oasisabm.activity import Schedule
from oasisabm.settings import DESIRED_TIMES, DEFAULT_MODES, DEFAULT_MH_PARAMS, DEFAULT_OPERATORS, DEFAULT_VARIABLES, DEFAULT_ACTIVITIES, DEFAULT_P_OPERATORS

from typing import List, Dict, Tuple, Optional, Union


warnings.filterwarnings('once')





class ChoiceSetGenerator():
    """
    This class is used to generate choice sets of Schedule objects for given individuals.

    Attributes:
    -------------------
    - schedules: List of Schedule objects
    - param_file: location of parameters for the target distribution (or dictionary of parameters)
    - n_alt: number of alternatives in the choice set
    - mh_params: dictionary containing parameters for the random walk
    - activities: list of activities
    - operators: list of operators
    - p_operators: probabilities of operators
    - modes: list of modes
    - locations: list of locations
    - variables: list of variables for target distribution
    - outfile: location of file to save result
    - desired_times: preferred start times and durations per activity type, {type: {'desired_start_time':..., 'desired_duration':...}}
    - max_episodes: maximum number of episodes per activity type (excluding the first and last home activities), see metropolis_hastings.is_valid
    - min_activities: minimum number of activities in sampled schedules
    - min_duration: minimum duration of each activity in sampled schedules, in hours
    - circular: if True, start time deviations wrap around midnight (original behaviour). Use False to match the simulation model.
    - seed: random seed. Each individual uses seed + index.
    - sample_in_walk: if True, the n_alt-1 alternatives are drawn from the states of the random walk in generate_set,
      which only returns those. This saves memory for large samples.



    Methods:
    ------------------
    - generate_set: generates choice set for a given individual.
    - run: Run metropolis_hastings algorithm for full dataset.
    - run: Run metropolis_hastings algorithm for full dataset, using parallel processing.
    - compute_sample_correction: Returns the corrective term for the utility function
    - train_test_sets: Creates train and test Dataframes to use in Biogeme.

    """
    def __init__(self, schedules:List, params_file:Union[str, Dict, None],n_alt:int = 10, mh_params:Dict = DEFAULT_MH_PARAMS,
     activities:Optional[List] = DEFAULT_ACTIVITIES, operators:Optional[List] = DEFAULT_OPERATORS, proba_operators:Optional[List] = None,
     modes: Optional[List] = DEFAULT_MODES, locations:Optional[List] = None, variables: List = DEFAULT_VARIABLES, outfile:Optional[str] ='choice_set.joblib', seed:int = 42,
     desired_times: Optional[Dict] = None, max_episodes: Optional[Dict] = None, min_activities: int = 1, min_duration: float = 0, circular: bool = True, home: str = 'home', sample_in_walk: bool = False, **kwargs):

        self.schedules = schedules
        self.params_file = params_file
        self.n_alt = n_alt
        self.mh_params = mh_params
        self.activities = activities
        self.operators = operators
        self.p_operators = proba_operators
        self.modes = modes
        self.outfile = outfile
        self.seed = seed

        self.locations = locations

        self.variables = variables
        self.desired_times = DESIRED_TIMES if desired_times is None else desired_times
        self.max_episodes = max_episodes
        self.min_activities = min_activities
        self.min_duration = min_duration
        self.sample_in_walk = sample_in_walk
        self.circular = circular
        self.home = home

        self.choice_sets = []
        self.accepted_operators = []
        self.acceptance_probas = []

        self.sample_corrections = []



    def generate_set(self, schedule: Schedule, seed: Optional[int] = None)-> Tuple[List, List, List]:
        """
        Generates choice set for a given individual.

        Parameters
        ----------
        - schedule : Schedule object
        - seed: random seed

        Returns
        ----------
        Choice sets, accepted operators and probabilities. The first element of the choice set is the chosen schedule,
        the others are the states of the random walk (after burn-in, every n_skip iterations), including repeated states.

        """
        if seed is not None:
            np.random.seed(seed)

        choice_set = [schedule]
        list_op = []
        probas = []

        n_skip = self.mh_params["n_skip"]
        n_burn = self.mh_params["n_burn"]

        if not self.locations:
            all_locations = schedule.all_locations
        else:
            all_locations = self.locations

        steps = mh.random_walk(init_sched = deepcopy(schedule), operators=self.operators, p_operators = self.p_operators,
        list_act=self.activities, list_loc = all_locations, list_modes = self.modes, param_file = self.params_file,
        max_episodes = self.max_episodes, min_activities = self.min_activities, min_duration = self.min_duration, yield_rejected = True, **self.mh_params)

        n = 0
        for state, op, pb in steps:
            if (n>n_burn) and (n%n_skip == 0):
                optype = op.optype
                if optype == "MetaOperator":
                    optype = op.meta_type

                list_op.append(optype)
                choice_set.append(state)
                probas.append(pb)

            n += 1

        if self.sample_in_walk and len(choice_set) > 1:
            idx = np.random.choice(range(1, len(choice_set)), self.n_alt - 1, replace = True)
            choice_set = [choice_set[0]] + [choice_set[i] for i in idx]
            list_op = [list_op[i-1] for i in idx]
            probas = [probas[i-1] for i in idx]

        return choice_set, list_op, probas

    def run(self) -> None:
        """
        Run metropolis_hastings algorithm for full dataset and saves choice sets, accepted operators and acceptance probabilities to file.
        """
        start = datetime.now()
        for i, schedule in enumerate(self.schedules):
            print(f"Starting generation for individual {i}.\n")

            choice_set, list_op, probas = self.generate_set(schedule, self.seed + i)

            self.choice_sets.append(choice_set)
            self.accepted_operators.append(list_op)
            self.acceptance_probas.append(probas)

        end = datetime.now()
        print(f"Total runtime: {end-start}")

        if self.outfile:
            joblib.dump([self.choice_sets,self.accepted_operators, self.acceptance_probas], self.outfile)


    def run_parallel(self, n_cpus:Optional[int] = None, verbose:int = 5)->None:
        """
        Run metropolis_hastings algorithm for full dataset using parallel processing. Saves choice sets, accepted operators and acceptance probabilities to file.

        Parameters
        ----------
        n_cpus: number of CPUs to use for the parallel process.
        verbose: gives frequency of progress ouptuts

        """

        if not n_cpus:
            n_cpus = cpu_count()

        delayed_output = [delayed(self.generate_set)(s, self.seed + i) for i, s in enumerate(self.schedules)]
        results = Parallel(n_jobs=n_cpus, verbose=verbose)(delayed_output)

        self.choice_sets, self.accepted_operators, self.acceptance_probas = [list(r) for r in zip(*results)]

        if self.outfile:
            joblib.dump([self.choice_sets,self.accepted_operators, self.acceptance_probas], self.outfile)


    def compute_sample_correction(self, counts: List, log_weights: List)->List:
        """
        Returns the corrective term for the utility function, to estimate the model on the sampled choice set passed as input.
        The choice set contains the chosen alternative and alternatives sampled from the target distribution q of the random walk.
        The correction for alternative j is ln(k_j / q(j)), where k_j is the number of times j is in the choice set
        (the chosen alternative counts once more). See Flötteröd and Bierlaire (2013), "Metropolis-Hastings sampling of paths",
        and Ben-Akiva & Lerman (1985) "Discrete choice analysis", p.266.

        Parameters
        ---------------
        - counts: for each individual, number of times each unique alternative is in the choice set
        - log_weights: for each individual, logarithm of the (unnormalised) target weight of each unique alternative
        """

        sample_correction = [list(np.log(np.array(k, dtype=float)) - np.array(lw, dtype=float)) for k, lw in zip(counts, log_weights)]
        return sample_correction

    def _log_weight(self, schedule: Schedule) -> float:
        """Logarithm of the target weight of a schedule (0 for a uniform target distribution).
        Use a fixed random term (mh_params['rnd_term'], e.g. 0) so that the weight is deterministic."""
        if self.mh_params.get('uniform', False) or self.params_file is None:
            return 0.0
        utility = mh.target_weight(schedule, param_file=self.params_file, rnd_term=self.mh_params.get('rnd_term'))
        if self.mh_params.get('log_weights', False):
            return float(utility)
        return float(np.log(abs(utility)))

    def schedule_features(self, schedule: Schedule) -> Dict:
        """
        Computes the variables of the utility function for a schedule, summed over the episodes of each activity type.

        Parameters
        --------------
        - schedule: Schedule object

        Returns
        --------------
        Dictionary of variables, with keys '{activity}:{variable}' where variable is one of count, participation,
        start_time (first episode), duration (total), early, late, short, long.
        For the home activity, only mid-day episodes are counted, and there are no timing variables.
        """
        features = {}
        middle = schedule.list_act[1:-1]

        for at in self.activities:
            episodes = [a for a in middle if a.label == at]
            count = len(episodes)
            starts = np.array([a.start_time for a in episodes], dtype=float)
            durations = np.array([a.duration for a in episodes], dtype=float)
            features[f'{at}:count'] = count
            features[f'{at}:participation'] = int(count > 0)
            features[f'{at}:start_time'] = starts.min() if count else 0
            features[f'{at}:duration'] = durations.sum() if count else 0

            if at in [self.home, 'dawn', 'dusk'] or at not in self.desired_times or count == 0:
                for v in ['early', 'late', 'short', 'long']:
                    features[f'{at}:{v}'] = 0
                continue

            st_diff = starts - self.desired_times[at]['desired_start_time']
            d_diff = durations - self.desired_times[at]['desired_duration']

            if self.circular:
                early = ((st_diff>=-12) & (st_diff<=0))*(-st_diff) + ((st_diff>=12) & (st_diff<=24))*(24-st_diff)
                late = ((st_diff>=0)&(st_diff<12))*(st_diff) + ((st_diff>=-24) & (st_diff<-12))*(24+st_diff)
            else:
                early = np.maximum(-st_diff, 0)
                late = np.maximum(st_diff, 0)

            features[f'{at}:early'] = float(np.sum(early))
            features[f'{at}:late'] = float(np.sum(late))
            features[f'{at}:short'] = float(np.sum(np.maximum(-d_diff, 0)))
            features[f'{at}:long'] = float(np.sum(np.maximum(d_diff, 0)))

        return features


    def train_test_sets(self, k:int=1, train_ratio:float= 0.7) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
        """
        Creates train and test Dataframes to use in Biogeme.

        For each individual, the choice set contains the chosen schedule and n_alt-1 schedules drawn (with replacement)
        from the states of the random walk. Repeated draws are merged into one alternative, so choice sets can have fewer
        than n_alt alternatives: the availability of each alternative is given in the 'av' column. The sampling correction
        is in the 'prob_corr' column (see compute_sample_correction). The chosen alternative always has alt_id 0.

        Parameters:
        --------------
        - k: proportionality constant for sample correction (kept for compatibility, it does not change the estimates)
        - train_ratio: train test split (default: 70% of observations will be used for the train set). Use 1 to estimate on all observations.

        Returns:
        ---------------
        - Train dataset in wide and long format, Test dataset (list of choice sets, as lists of pandas DataFrames, the first one being chosen)
        """

        n_obs = len(self.choice_sets)
        n_train = math.ceil(train_ratio*n_obs)
        order = np.random.permutation(n_obs)
        train_ids = sorted(order[:n_train])
        test_ids = sorted(order[n_train:])

        formatted_test = [[schedule_to_pandas(sched) for sched in self.choice_sets[t]] for t in test_ids]

        rows = []
        counts_all = []
        log_weights_all = []
        unique_all = []

        for obs_id, t in enumerate(train_ids):
            chosen = self.choice_sets[t][0]
            draws = self.choice_sets[t][1:]

            #count unique alternatives: the chosen alternative, plus n_alt-1 draws from the random walk
            counts = {chosen: 1}
            if draws:
                if self.sample_in_walk:
                    sampled = draws #already drawn in generate_set
                else:
                    sampled = [draws[idx] for idx in np.random.choice(len(draws), self.n_alt - 1, replace = True)]
                for sched in sampled:
                    counts[sched] = counts.get(sched, 0) + 1

            unique = list(counts.keys())
            counts_all.append([counts[u] for u in unique])
            log_weights_all.append([self._log_weight(u) for u in unique])
            unique_all.append(unique)

        self.sample_corrections = self.compute_sample_correction(counts_all, log_weights_all)

        for obs_id, unique in enumerate(unique_all):
            for alt_id, sched in enumerate(unique):
                row = {'obs_id': obs_id, 'alt_id': alt_id, 'choice': int(alt_id == 0), 'av': 1,
                       'prob_corr': self.sample_corrections[obs_id][alt_id]}
                row.update(self.schedule_features(sched))
                rows.append(row)

        df_long = pd.DataFrame(rows)

        #Convert from long to wide: one row per observation, with columns '{variable}_{alt_id}'. Missing alternatives are unavailable.
        df_long_full = df_long.set_index(['obs_id', 'alt_id']).reindex(
            pd.MultiIndex.from_product([range(len(unique_all)), range(self.n_alt)], names=['obs_id', 'alt_id']), fill_value=0)
        df_wide = df_long_full.drop(columns='choice').unstack('alt_id')
        df_wide.columns = [f'{c}_{i}' for c, i in df_wide.columns]
        df_wide = df_wide.reset_index()
        df_wide['choice'] = 0

        return df_wide, df_long, formatted_test
