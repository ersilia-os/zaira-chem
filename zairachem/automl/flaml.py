import os
import flaml
from ..tools.ghost.ghost import Ghost

DEFAULT_TIME_BUDGET_MIN = 10


class FlamlClassifier(object):

    def __init__(self, metric="auto"):
        self.task = "classification"
        self.metric = metric

    def fit(self, time_budget=DEFAULT_TIME_BUDGET_MIN):
        automl_settings = {
            "time_budget": int(time_budget) * 60,  #  in seconds
            "metric": self.metric,
            "task": self.task,
            "log_file_name": "automl.log",
            "verbose": 0,
        }
        automl = AutoML()
        automl.fit(X_train=X, y_train=y, **automl_settings)
        self.mdl = automl.model
        self.meta = {
            "estimator": automl.best_estimator,
            "loss": automl.best_loss,
            "metric": self.metric,
            "task": self.task,
        }
        # remove catboost info folder if generated
        cwd = os.getcwd()
        catboost_info = os.path.join(cwd, "catboost_info")
        if os.path.exists(catboost_info):
            shutil.rmtree(catboost_info)
        if os.path.exists("automl.log"):
            os.remove("automl.log")
        self.ghost = Ghost(self.mdl)
        self.ghost.get_threshold(X, y)

    def save(self):
        pass

    def load(self):
        pass

    def predict_proba(self, X):
        return self.automl.predict_proba(X)

    def predict(self, X):
        return self.ghost.predict(X)


class FlamlRegressor(object):

    def __init__(self):
        self.task = "regression"

    def fit(self):
        pass

    def predict(self):
        pass
