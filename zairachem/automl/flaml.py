import os
import numpy as np
import uuid
import shutil
import joblib
import json
import collections
from sklearn.model_selection import KFold, StratifiedKFold

from flaml import AutoML
from ..tools.ghost.ghost import GhostLight

from ..vars import N_FOLDS


FLAML_TIME_BUDGET_SECONDS = 60

FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS = 30 # 30
FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS = 600 # 600
FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS = 10 # 10
FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS = 60 # 60

FLAML_COLD_MAXIMUM_ITERATIONS = 1000
FLAML_WARM_MAXIMUM_ITERATIONS = 100

MIN_SAMPLES_TO_ALLOW_EVALUATION = 1

SPLITTING_ROUNDS = 3


class Splitter(object):
    def __init__(self, X, y, is_clf):
        if is_clf:
            self.splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
        else:
            self.splitter = KFold(n_splits=N_FOLDS, shuffle=True)
        self.X = X
        self.y = y

    def split(self):
        for _ in range(SPLITTING_ROUNDS):
            for tr_idxs, te_idxs in self.splitter.split(self.X, self.y):
                yield tr_idxs, te_idxs


class FlamlSettings(object):
    def __init__(self, y):
        # TODO: This will help elucidate best metric, or use multitasking if necessary
        self.y = np.array(y)
        self.is_clf = self._is_binary()

    def _is_binary(self):
        if len(set(self.y)) == 2:
            return True
        else:
            return False

    def _has_enough_samples_per_group(self, groups):
        if not self.is_clf:
            return True
        group_idxs = collections.defaultdict(list)
        for i, g in enumerate(groups):
            group_idxs[g] += [i]
        for k, v in group_idxs.items():
            v = np.array(v)
            y_ = self.y[v]
            n_0 = np.sum(y_ == 0)
            n_1 = np.sum(y_ == 1)
            if (
                n_0 < MIN_SAMPLES_TO_ALLOW_EVALUATION
                or n_1 < MIN_SAMPLES_TO_ALLOW_EVALUATION
            ):
                return False
        return True

    def cast_groups(self, groups):
        if groups is not None:
            if self._has_enough_samples_per_group(groups):
                return groups
        groups = np.zeros(len(self.y), dtype=int)
        if self.is_clf:
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True)
            i = 0
            for _, test_idx in splitter.split(self.y, self.y):
                groups[test_idx] = i
                i += 1
        else:
            splitter = KFold(n_splits=N_FOLDS, shuffle=True)
            i = 0
            for _, test_idx in splitter.split(self.y):
                groups[test_idx] = i
                i += 1
        return np.array([int(x) for x in groups])

    def get_metric(self, task):
        # TODO: Adapt to imbalanced learning
        if task == "classification":
            return "auto"
        else:
            return "auto"

    def get_automl_settings(self, task, time_budget, estimators, groups):
        automl_settings = {
            "time_budget": int(
                np.clip(
                    time_budget,
                    FLAML_COLD_MINIMUM_TIME_BUDGET_SECONDS,
                    FLAML_COLD_MAXIMUM_TIME_BUDGET_SECONDS,
                )
            ),
            "metric": self.get_metric(task),
            "task": task,
            "log_file_name": "automl.log",
            "log_training_metric": True,
            "verbose": 3,
            "early_stop": True,
            "max_iter": int(min(
                len(self.y)/3, FLAML_COLD_MAXIMUM_ITERATIONS
            )),  # TODO better heuristic based on sample size
        }
        if estimators is not None:
            automl_settings["estimator_list"] = estimators
        groups = self.cast_groups(groups)
        automl_settings["split_type"] = "group"
        automl_settings["groups"] = groups
        return automl_settings


class FlamlEstimator(object):
    def __init__(self, task, name=None, fit_with_groups=False):
        if name is None:
            name = str(uuid.uuid4())
        self.name = name
        self.task = task
        self.main_mdl = None
        if self.task == "classification":
            self.is_clf = True
        else:
            self.is_clf = False
        self._fit_with_groups = fit_with_groups

    def _clean_log(self, automl_settings):
        cwd = os.getcwd()
        log_file = os.path.join(cwd, automl_settings["log_file_name"])
        if os.path.exists(log_file):
            os.remove(log_file)
        # remove catboost info folder if generated
        catboost_info = os.path.join(cwd, "catboost_info")
        if os.path.exists(catboost_info):
            shutil.rmtree(catboost_info)

    def fit_main(self, X, y, time_budget, estimators, groups):
        automl_settings = FlamlSettings(y).get_automl_settings(
            task=self.task,
            time_budget=time_budget,
            estimators=estimators,
            groups=groups,
        )
        self.main_groups = groups
        self.main_mdl = AutoML()
        _automl_settings = dict((k, v) for k, v in automl_settings.items())
        if not self._fit_with_groups:
            _automl_settings["eval_method"] = "auto"
            _automl_settings["split_type"] = None
            _automl_settings["groups"] = None
        print(_automl_settings)
        self.main_mdl.fit(X_train=X, y_train=y, **_automl_settings)
        self.main_automl_settings = automl_settings
        if self.is_clf:
            y_hat = self.main_mdl.predict_proba(X)[:, 1]
        else:
            y_hat = self.main_mdl.predict(X)
        return y_hat

    def fit_predict_out_of_sample(self, X, y):
        assert self.main_mdl is not None
        automl_settings = self.main_automl_settings.copy()
        automl_settings["time_budget"] = int(
            np.clip(
                automl_settings["time_budget"] * 0.1,
                FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS,
                FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS,
            )
        )
        y = np.array(y)
        best_models = self.main_mdl.best_config_per_estimator
        best_estimator = self.main_mdl.best_estimator
        starting_point = {best_estimator: best_models[best_estimator]}
        print(starting_point)
        splitter = Splitter(X, y, is_clf=self.is_clf)
        k = 0
        results = collections.defaultdict(list)
        for tr_idxs, te_idxs in splitter.split():
            tag = "oos_{0}".format(k)
            automl_settings["log_file_name"] = "{0}_automl.log".format(tag)
            X_tr = X[tr_idxs]
            X_te = X[te_idxs]
            y_tr = y[tr_idxs]
            automl_settings["eval_method"] = "auto"
            automl_settings["split_type"] = None
            automl_settings["groups"] = None
            automl_settings["estimator_list"] = [best_estimator]
            automl_settings["max_iter"] = min(
                int(self.main_automl_settings["max_iter"] * 0.25) + 1,
                FLAML_WARM_MAXIMUM_ITERATIONS,
            )
            model = AutoML()
            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                #starting_points=starting_point,
                **automl_settings
            )
            if self.is_clf:
                y_te_hat = model.predict_proba(X_te)[:, 1]
            else:
                y_te_hat = model.predict(X_te)
            for i, idx in enumerate(te_idxs):
                results[idx] += [y_te_hat[i]]
            self._clean_log(automl_settings)
            k += 1
        y_hat = []
        for i in range(len(y)):
            y_hat += [np.mean(results[i])]
        return np.array(y_hat)

    def fit_predict_by_group(self, X, y):
        assert self.main_mdl is not None
        automl_settings = self.main_automl_settings.copy()
        y = np.array(y)
        results = collections.OrderedDict()
        groups = np.array(automl_settings["groups"])
        automl_settings["time_budget"] = int(
            np.clip(
                automl_settings["time_budget"] * 0.1,
                FLAML_WARM_MINIMUM_TIME_BUDGET_SECONDS,
                FLAML_WARM_MAXIMUM_TIME_BUDGET_SECONDS,
            )
        )
        folds = sorted(set(groups))
        best_models = self.main_mdl.best_config_per_estimator
        best_estimator = self.main_mdl.best_estimator
        starting_point = {best_estimator: best_models[best_estimator]}
        for fold in folds:
            tag = "fold_{0}".format(fold)
            automl_settings["log_file_name"] = "{0}_automl.log".format(tag)
            tr_idxs = []
            te_idxs = []
            for i, g in enumerate(groups):
                if g != fold:
                    tr_idxs += [i]
                else:
                    te_idxs += [i]
            tr_idxs = np.array(tr_idxs)
            te_idxs = np.array(te_idxs)
            X_tr = X[tr_idxs]
            X_te = X[te_idxs]
            y_tr = y[tr_idxs]
            y_te = y[te_idxs]
            if self._fit_with_groups:
                groups_tr = groups[tr_idxs]
                unique_groups = list(set(groups_tr))
                groups_mapping = {}
                for i, g in enumerate(unique_groups):
                    groups_mapping[g] = i
                automl_settings["groups"] = np.array(
                    [groups_mapping[g] for g in groups_tr]
                )
                automl_settings["n_splits"] = len(unique_groups)
            else:
                automl_settings["eval_method"] = "auto"
                automl_settings["split_type"] = None
                automl_settings["groups"] = None
            automl_settings["estimator_list"] = [best_estimator]
            automl_settings["max_iter"] = min(
                int(self.main_automl_settings["max_iter"] * 0.25) + 1,
                FLAML_WARM_MAXIMUM_ITERATIONS,
            )
            model = AutoML()
            print(automl_settings)
            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                # starting_points=starting_point,
                **automl_settings
            )
            if self.is_clf:
                y_te_hat = model.predict_proba(X_te)[:, 1]
            else:
                y_te_hat = model.predict(X_te)
            self._clean_log(automl_settings)
            results[tag] = {"idxs": te_idxs, "y": y_te, "y_hat": y_te_hat}
        return results

    def fit_predict(self, X, y, time_budget, estimators, groups, predict_by_group):
        self.fit_main(X, y, time_budget, estimators, groups)
        y_hat_main = self.fit_predict_out_of_sample(X, y)
        if predict_by_group:
            group_results = self.fit_predict_by_group(X, y)
        else:
            group_results = {}
        self._clean_log(self.main_automl_settings)
        results = collections.OrderedDict()
        results["main"] = {"idxs": None, "y": y, "y_hat": y_hat_main}
        for k, v in group_results.items():
            results[k] = v
        return results


class Binarizer(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def binarize(self, y_hat):
        y_bin = []
        for y in y_hat:
            if y > self.threshold:
                y_bin += [1]
            else:
                y_bin += [0]
        return np.array(y_bin, dtype=np.uint8)


class FlamlClassifier(object):
    def __init__(self, name=None):
        self.task = "classification"
        self.estimator = FlamlEstimator(task=self.task, name=name)
        self.name = self.estimator.name

    def fit(
        self,
        X,
        y,
        time_budget=FLAML_TIME_BUDGET_SECONDS,
        estimators=None,
        groups=None,
        predict_by_group=False,
    ):
        self.results = self.estimator.fit_predict(
            X, y, time_budget, estimators, groups, predict_by_group=predict_by_group
        )
        for k, v in self.results.items():
            threshold = GhostLight().get_threshold(
                self.results[k]["y"], self.results[k]["y_hat"]
            )
            if k == "main":
                self.threshold = threshold
            binarizer = Binarizer(threshold)
            self.results[k]["b_hat"] = binarizer.binarize(v["y_hat"])

    def save(self, file_name):
        joblib.dump(self.estimator.main_mdl, file_name)
        data = {"threshold": self.threshold}
        with open(file_name.split(".")[0] + ".json", "w") as f:
            json.dump(data, f)

    def load(self, file_name):
        model = joblib.load(file_name)
        with open(file_name.split(".")[0] + ".json", "r") as f:
            data = json.load(f)
        threshold = data["threshold"]
        return FlamlClassifierArtifact(model, threshold)


class FlamlClassifierArtifact(object):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        self.binarizer = Binarizer(self.threshold)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        y_hat = self.predict_proba(X)
        y_bin = self.binarizer.binarize(y_hat)
        return y_bin

    def run(self, X, y=None):
        results = collections.OrderedDict()
        results["main"] = {
            "idxs": None,
            "y": y,
            "y_hat": self.predict_proba(X),
            "b_hat": self.predict(X),
        }
        return results


class FlamlRegressor(object):
    def __init__(self, name=None):
        self.task = "regression"
        self.estimator = FlamlEstimator(task=self.task, name=name)
        self.name = self.estimator.name

    def fit(
        self,
        X,
        y,
        time_budget=FLAML_TIME_BUDGET_SECONDS,
        estimators=None,
        groups=None,
        predict_by_group=False,
    ):
        self.results = self.estimator.fit_predict(
            X, y, time_budget, estimators, groups, predict_by_group=predict_by_group
        )

    def save(self, file_name):
        joblib.dump(self.estimator.main_mdl, file_name)
        data = {}
        with open(file_name.split(".")[0] + ".json", "w") as f:
            json.dump(data, f)

    def load(self, file_name):
        model = joblib.load(file_name)
        return FlamlRegressorArtifact(model)


class FlamlRegressorArtifact(object):
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def run(self, X, y=None):
        results = collections.OrderedDict()
        results["main"] = {"idxs": None, "y": y, "y_hat": self.predict(X)}
        return results
