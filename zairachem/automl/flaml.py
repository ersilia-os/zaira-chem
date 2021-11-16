import os
import numpy as np
import uuid
from sklearn.model_selection import KFold

from flaml import AutoML
from ..tools.ghost.ghost import Ghost

from ..vars import N_FOLDS


DEFAULT_TIME_BUDGET_MIN = 0.1


class FlamlSettings(object):
    def __init__(self, y):
        # TODO: This will help elucidate best metric, or use multitasking if necessary
        self.y = np.array(y)

    def cast_groups(self, groups):
        if groups is not None:
            return groups
        else:
            splitter = KFold(n_splits=N_FOLDS, shuffle=True)
            groups = np.zeros(len(self.y), dtype=int)
            i = 0
            for _, test_idx in splitter.split(folds):
                groups[test_idx] = i
                i += 1
            return list(groups)

    def get_metric(self, task):
        # TODO: Adapt to imbalanced learning
        if task == "classification":
            return "auto"
        else:
            return "auto"

    def get_automl_settings(self, task, time_budget, estimators, groups):
        automl_settings = {
            "time_budget": int(time_budget * 60) + 1,  #  in seconds
            "metric": get_metric(task),
            "task": task,
            "log_file_name": "automl.log",
            "verbose": 3,
        }
        if estimators is not None:
            automl_settings["estimator_list"] = estimators
        groups = self.cast_groups(groups)
        automl_settings["split_type"] = "group"
        automl_settings["groups"] = groups
        return automl_settings


class FlamlEstimator(object):
    def __init__(self, task, name=None):
        if name is None:
            name = list(uuid.uuid4())
        self.name = name
        self.task = task
        self.main_mdl = None
        if self.task == "classification":
            self.is_clf = True
        else:
            self.is_clf = False

    def _clean_log(self, automl_settings):
        log_file = automl_settings["log_file_name"]
        # remove catboost info folder if generated
        cwd = os.getcwd()
        catboost_info = os.path.join(cwd, "catboost_info")
        if os.path.exists(catboost_info):
            shutil.rmtree(catboost_info)
        if os.path.exists(log_file):
            os.remove(log_file)

    def fit_main(
        self, X, y, time_budget=DEFAULT_TIME_BUDGET_MIN, estimators=None, groups=None
    ):
        automl_settings = FlamlSettings(y).get_automl_settings(
            task=self.task,
            time_budget=time_budget,
            estimators=estimators,
            groups=groups,
        )
        self.main_groups = groups
        self.main_mdl = AutoML()
        self.main_mdl.fit(X_train=X, y_train=y, **automl_settings)
        self.main_automl_settings = automl_settings

    def fit_predict_by_group(self, X, y):
        assert self.main_mdl is not None
        automl_settings = self.main_automl_settings
        y = np.array(y)
        y_hat = np.zeros(y.shape)
        groups = np.array(automl_settings["groups"])
        automl_settings["time_budget"] = 60  # TODO
        folds = sorted(set(groups))
        for fold in folds:
            automl_settings[
                "log_file_name"
            ] = "automl_group.log"  # USE LOGS CORRESPONDINGLY
            tag = "fold_{0}".format(fold)
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
            model = AutoML()
            model.retrain_from_log(
                log_file_name=self.main_automl_settings["log_file_name"],
                X_train=X_tr,
                y_tr=y_tr,
                **automl_settings
            )
            if self._is_clf:
                y_te_hat = model.predict_proba(X_te)[:, 1]
            else:
                y_te_hat = model.predict(X_te)
            self._clean_log(automl_settings)
            y_hat[te_idxs] = y_te_hat
        return y_hat

    def fit_predict(self, X, y):
        self.fit_main(X, y)
        y_hat = self.fit_predict_by_group(X, y)
        self._clean_log(self.main_automl_settings)
        return y_hat


class FlamlClassifier(object):
    def __init__(self, name=None):
        if name is None:
            name = list(uuid.uuid4())
        self.name = name
        self.task = "classification"

    def fit(
        self, X, y, time_budget=DEFAULT_TIME_BUDGET_MIN, estimators=None, groups=None
    ):
        automl_settings = FlamlSettings().get_automl_settings(
            task=self.task,
            time_budget=time_budget,
            estimators=estimators,
            groups=groups,
        )
        self.mdl = AutoML()
        self.mdl.fit(X_train=X, y_train=y, **automl_settings)
        # remove catboost info folder if generated
        cwd = os.getcwd()
        catboost_info = os.path.join(cwd, "catboost_info")
        if os.path.exists(catboost_info):
            shutil.rmtree(catboost_info)
        if os.path.exists("automl.log"):
            os.remove("automl.log")
        self.ghost = Ghost(self.mdl.model)
        self.ghost.get_threshold(X, y)

    def predict_proba(self, X):
        return self.mdl.predict_proba(X)

    def predict(self, X):
        return self.ghost.predict(X)

    def save(self):
        pass

    def load(self):
        pass


class FlamlRegressor(object):
    def __init__(self):
        self.task = "regression"

    def fit(
        self, X, y, time_budget=DEFAULT_TIME_BUDGET_MIN, estimators=None, groups=None
    ):
        automl_settings = get_automl_settings(
            metric=self.metric,
            task=self.task,
            time_budget=time_budget,
            estimators=estimators,
            groups=groups,
        )
        self.mdl = AutoML()
        self.mdl.fit(X_train=X, y_train=y, **automl_settings)

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
