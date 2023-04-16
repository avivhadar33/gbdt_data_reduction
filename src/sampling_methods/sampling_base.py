import numpy as np
import matplotlib.pyplot as plt
import tqdm
import src.sampling_methods.sampling_utils as sampling_utils
from src.dataset import Dataset
from abc import ABC
from copy import deepcopy
import xgboost as xgb
import random


class Experiment(ABC):

    def __init__(self, name, model=None):

        self.name = name
        self.model = model if model else xgb.XGBClassifier(random_state=random.randint(1, 30))

        self.pct = None
        self.scores = None
        self.times = None
        self.stds = None
        self.mins = None
        self.maxs = None
        self.trials_number = None

    def sample_func(self, dataset: Dataset, p):
        pass

    def test_sample_method(self,
                           dataset: Dataset,
                           percents=sampling_utils.DEFAULT_PCT,
                           trials_number=1,
                           update_pcts=True,
                           print_results=False):

        self.pct = deepcopy(percents) if not update_pcts else []
        self.scores = []
        self.times = []
        self.stds = []
        self.mins = []
        self.maxs = []
        self.trials_number = trials_number

        for p in tqdm.tqdm(percents):
            scores = []
            times = []
            pcts = []
            for i in range(self.trials_number):
                results = self.sample_func(dataset, p)
                if print_results:
                    print(results)
                score, cur_time = results[0], results[1]
                if update_pcts:
                    pcts.append(results[2])
                scores.append(score)
                times.append(cur_time)

            if update_pcts:
                self.pct.append(sum(pcts) / self.trials_number)
            self.scores.append(sum(scores) / self.trials_number)
            self.times.append(sum(times) / self.trials_number)
            self.stds.append(np.std(scores))
            self.mins.append(min(scores))
            self.maxs.append(max(scores))

    def print_scores(self, color='r'):
        if self.trials_number > 1:
            plt.errorbar(self.pct, self.scores, yerr=self.stds, fmt='bo', color=color, label=self.name, capsize=4)
            plt.plot(self.pct, self.mins, '.', color=color)
            plt.plot(self.pct, self.scores, '-', color=color)
        else:
            plt.plot(self.pct, self.scores, '.', color=color, label=self.name)
            plt.plot(self.pct, self.scores, '-', color=color)

    def print_times(self, color='r'):
        plt.plot(self.pct, self.times, 'bo', color=color, label=self.name)
