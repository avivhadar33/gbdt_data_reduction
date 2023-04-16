import time
import xgboost as xgb
import random
from sklearn.tree import DecisionTreeClassifier

from src.sampling_methods.sampling_base import Experiment
import src.sampling_methods.sampling_utils as sampling_utils
from src.dataset import Dataset
from src.gbdt_model_paths.model_path import ModelPath


class RandomSample(Experiment):

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        dataset.X_train[dataset.target_col] = dataset.y_train
        X_real_train = dataset.X_train.groupby(dataset.target_col).sample(frac=p)
        y_real_train = dataset.y_train[X_real_train.index]

        dataset.X_train.drop(columns=dataset.target_col, inplace=True)
        X_real_train.drop(columns=dataset.target_col, inplace=True)

        self.model.fit(X_real_train, y_real_train)
        return dataset.metric(dataset.y_test, self.model.predict(dataset.X_test)),\
               time.time() - s,\
               len(X_real_train) / len(dataset.X_train)


class XgboostSubsample(Experiment):

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        if self.model == sampling_utils.REGRESSION:
            model = xgb.XGBRegressor(subsample=p, random_state=random.randint(1, 30))
        else:
            model = xgb.XGBClassifier(subsample=p, random_state=random.randint(1, 30))
        X_real_train = dataset.X_train
        y_real_train = dataset.y_train[X_real_train.index]
        model.fit(X_real_train, y_real_train)
        return dataset.metric(dataset.y_test, model.predict(dataset.X_test)), time.time() - s


class TreeSample(Experiment):

    def sample_func(self, dataset: Dataset, p):
        s = time.time()
        singel_tree_model = DecisionTreeClassifier(class_weight='balanced')
        singel_tree_model.fit(dataset.X_train.fillna(0), dataset.y_train)

        leaf_id = singel_tree_model.apply(dataset.X_train.fillna(0))
        dataset.X_train['leaf_id'] = leaf_id
        dataset.X_train[dataset.target_col] = dataset.y_train
        X_real_train = sampling_utils.sample_from_groups_round_up(
            dataset.X_train.groupby(['leaf_id', dataset.target_col],as_index=False), p)
        y_real_train = dataset.y_train[X_real_train.index]

        dataset.X_train.drop(columns=['leaf_id', dataset.target_col], inplace=True)
        X_real_train.drop(columns=['leaf_id', dataset.target_col], inplace=True)

        self.model.fit(X_real_train, y_real_train)
        return dataset.metric(dataset.y_test, self.model.predict(dataset.X_test)),\
               time.time() - s,\
               len(X_real_train) / len(dataset.X_train)


class CostumeSample(Experiment):
    def __init__(self, name, sample_functions, weights=None, model=None):
        super().__init__(name, model)
        self.sample_functions = sample_functions
        self.weights = weights

    def sample_func(self, dataset: Dataset, p):

        s = time.time()
        X_real_train, y_real_train = self.sample_functions(dataset, p)
        if not self.weights:
            self.model.fit(X_real_train, y_real_train)
        else:
            if callable(self.weights):
                self.model.fit(X_real_train, y_real_train, sample_weight=self.weights(X_real_train))
            else:
                self.model.fit(X_real_train, y_real_train, sample_weight=self.weights)
        return dataset.metric(dataset.y_test, self.model.predict(dataset.X_test)),\
               time.time() - s,\
               len(X_real_train) / len(dataset.X_train)


class XgboostPathSample(Experiment):

    def __init__(self, name, model_path: ModelPath, trees_to_use=None, model=None, use_target=True, index_name='index'):
        super().__init__(name, model)
        self.model_path = model_path
        self.trees_to_use = trees_to_use if trees_to_use else model_path.num_trees
        if use_target:
            self.groupby_columns = [f'diff_{i}' for i in range(self.trees_to_use)] + [model_path.datset.target_col]
        else:
            self.groupby_columns = [f'diff_{i}' for i in range(self.trees_to_use)]
        self.leaves_groupes = model_path.X_diff.groupby(self.groupby_columns, as_index=False)
        self.index_name = index_name

    def sample_func(self, dataset: Dataset, p):
        s = time.time()

        sampled = sampling_utils.sample_from_groups_round_up(self.leaves_groupes, p)
        X_real_train = dataset.X_train.loc[sampled.index]
        y_real_train = dataset.y_train.loc[sampled.index]

        # Get the weights
        self.model_path.X_diff['joined'] = self.model_path.X_diff.apply(
            lambda row: '_'.join(self.groupby_columns), axis=1)
        weight = self.model_path.X_diff.groupby('joined').count()[[dataset.target_col]].rename(
            columns={dataset.target_col: 'weight'}).reset_index()
        sampled_count = self.model_path.X_diff.loc[sampled.index, :].groupby('joined').count()[[dataset.target_col]].rename(
            columns={dataset.target_col: 'sampled_weight'}).reset_index()
        merged = self.model_path.X_diff[['joined']].reset_index()\
            .merge(weight, on='joined').\
            merge(sampled_count, on='joined').set_index(self.index_name)
        merged['normed_weight'] = merged.apply(lambda row: row['weight'] / row['sampled_weight'], axis=1)
        normed_weight = merged.loc[sampled.index, 'normed_weight']
        self.model_path.X_diff.drop(columns=['joined'], inplace=True)

        self.model.fit(X_real_train, y_real_train, sample_weight=normed_weight)
        return dataset.metric(dataset.y_test, self.model.predict(dataset.X_test)), \
               time.time() - s, \
               len(X_real_train) / len(dataset.X_train)














