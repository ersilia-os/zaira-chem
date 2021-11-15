import os
from flaml import AutoML
from ..tools.ghost.ghost import Ghost

DEFAULT_TIME_BUDGET_MIN = 1


def get_automl_settings(metric, task, time_budget, estimators=None, groups=None):
    automl_settings = {
        "time_budget": int(time_budget) * 60,  #  in seconds
        "metric": metric,
        "task": task,
        "log_file_name": "automl.log",
        "verbose": 3,
    }
    if estimators is not None:
        automl_settings["estimator_list"] = estimators
    if groups is not None:
        automl_settings["split_type"] = "group"
        automl_settings["groups"] = groups
    return automl_settings


class FlamlClassifier(object):
    def __init__(self, metric="auto"):
        self.task = "classification"
        self.metric = metric

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
