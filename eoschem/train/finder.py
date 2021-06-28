import numpy as np
import uuid
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
import mlflow
from sklearn import ensemble
from .evaluation import validation_score

# Others

SEED = 42
MIN_TIMEOUT = 60
N_ITER = 5

# Search configs

clf_configs ={
    "model": ensemble.RandomForestClassifier(class_weight="balanced", random_state=SEED),
    "params": {
        "n_estimators": [100, 500],
        "max_depth": [5, 10],
        "min_samples_split": [2, 3, 10],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2"]
        }
    }

reg_configs ={
    "model": ensemble.RandomForestRegressor(random_state=SEED),
    "params": {
        "n_estimators":[100, 500],
        "max_depth": [5, 10],
        "min_samples_split": [2, 3, 10],
        "max_features": ["sqrt", "log2"]
        }
    }


class ModelFinder(object):

    def __init__(self, is_clf, n_jobs=-1, n_iter=N_ITER, timeout=60):

        self.is_clf = is_clf
        if self.is_clf:
            search_configs = clf_configs
        else:
            search_configs = reg_configs
        self.params   = search_configs["params"]
        self.base_mod = search_configs["model"]

        self.n_jobs  = n_jobs
        self.n_iter  = n_iter
        timeout = timeout / n_iter
        timeout = np.max([MIN_TIMEOUT, timeout])
        self.timeout = timeout

        self.base_mod.set_params(n_jobs=n_jobs)

    def params2choices(self):
        params = {}
        for k,v in self.params.items():
            params[k] = hp.choice(k, v)
        return params

    def search(self, X, y):

        def objective(params):
            self.base_mod.set_params(**params)
            accuracy = validation_score(mod = self.base_mod,
                                        X = X, y = y,
                                        is_clf = self.is_clf
                                        )
            return {"loss": -accuracy, "status": STATUS_OK}

        params = self.params2choices()
        algo = tpe.suggest
        with mlflow.start_run():
            choice = fmin(
                fn=objective,
                space=params,
                algo=algo,
                max_evals=self.n_iter,
                rstate=np.random.RandomState(SEED))
        best_params = {}
        for k, idx in choice.items():
            best_params[k] = self.params[k][idx]
        return best_params

    def find_model(self, X, y):
        best_params = self.search(X, y)
        mod = self.base_mod
        mod.set_params(**best_params)
        return mod, best_params
