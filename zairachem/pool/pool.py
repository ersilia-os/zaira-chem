import os
import json
import joblib
import numpy as np

from flaml import AutoML
from .. import logger

from ..vars import (
    DATA_SUBFOLDER,
    MODELS_SUBFOLDER,
    POOL_SUBFOLDER,
    _CONFIG_FILENAME,
)

from ..metrics.metrics import Metric

from sklearn.linear_model import LinearRegression as Regressor
from sklearn.linear_model import LogisticRegressionCV as Classifier

_ESTIMATORS_FILENAME = "estimators.json"
_META_TRANSFORMER_FILENAME = "meta.pkl"
_META_MODEL_FILENAME = "model.pkl"

#  TODO: If multiple batches are available, balance selected models per batch
MAX_ESTIMATORS = 100

TIME_BUDGET = 1


class PoolEstimators(object):
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)

    def _find_estimators(self):
        logger.debug("Finding estimators")
        models_dir = os.path.join(self.dir, MODELS_SUBFOLDER)
        for batch in os.listdir(models_dir):
            if batch[:5] != "batch":
                continue
            for descriptor in os.listdir(os.path.join(models_dir, batch)):
                for split in os.listdir(os.path.join(models_dir, batch, descriptor)):
                    if split[:5] != "split":
                        continue
                    with open(
                        os.path.join(models_dir, batch, descriptor, split, "eval.json"),
                        "r",
                    ) as f:
                        eval = json.load(f)
                        score = eval["score"]
                        yield (batch, descriptor, split, score)

    def _select_estimators(self):
        logger.debug("Selecting estimators")
        estimators = {}
        for batch, descriptor, split, score in self._find_estimators():
            estimators[(batch, descriptor, split)] = score
        estimators = sorted(estimators.items(), key=lambda item: -item[1])[
            :MAX_ESTIMATORS
        ]
        with open(
            os.path.join(self.dir, POOL_SUBFOLDER, _ESTIMATORS_FILENAME), "w"
        ) as f:
            json.dump(estimators, f, indent=4)
        return estimators

    def get_preds(self):
        estimators = self._select_estimators()
        for estimator, score in estimators:
            logger.debug("Select estimator {0}".format(estimator))
            dir_ = os.path.join(
                self.dir, MODELS_SUBFOLDER, estimator[0], estimator[1], estimator[2]
            )
            y_pred_file = os.path.join(dir_, "y_pred.npy")
            with open(y_pred_file, "rb") as f:
                y_pred = np.load(f)
            dir_ = os.path.join(
                self.dir, DATA_SUBFOLDER, estimator[0], "splits", estimator[2]
            )
            idxs_file = os.path.join(dir_, "test_idx.npy")
            with open(idxs_file, "rb") as f:
                idxs = np.load(f)
            res = {
                "estimator": estimator,
                "idxs": idxs,
                "y_pred": y_pred,
                "score": score,
            }
            yield res


# TODO: Include molecule information (for example, region of the chemical space)


class MetaTransformer(object):
    def __init__(self, scores):
        self.scores = scores
        self.X = None

    def _avg_w(self):
        mask = ~np.isnan(self.X)
        a = []
        for i in range(self.X.shape[0]):
            v = self.X[i, mask[i]]
            w = self.scores[mask[i]] + 1e-6
            a += [np.average(v, weights=w)]
        return np.array(a)

    def _avg(self):
        return np.nanmean(self.X, axis=1)

    def _std(self):
        return np.nanstd(self.X, axis=1)

    def _max(self):
        return np.nanmax(self.X, axis=1)

    def _min(self):
        return np.nanmin(self.X, axis=1)

    def _funcs(self):
        funcs = [self._avg_w, self._avg, self._std, self._max, self._min]
        return funcs

    def transform(self, X):
        self.X = X
        funcs = self._funcs()
        X = np.zeros((X.shape[0], len(funcs)))
        for i, func in enumerate(funcs):
            X[:, i] = func()
        self.X = None
        return X


class MetaModel(object):
    def __init__(self, is_clf):
        self.is_clf = is_clf
        if self.is_clf:
            self.mdl = Classifier()
        else:
            self.mdl = Regressor()

    def fit(self, X, y):
        self.mdl.fit(X, y)


class Pool(object):
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.pool_estimators = PoolEstimators(self.dir)
        self.is_clf = self._is_classification()

    def _is_classification(self):
        with open(os.path.join(self.dir, DATA_SUBFOLDER, _CONFIG_FILENAME), "r") as f:
            config = json.load(f)
        return config["is_clf"]

    def _get_X_y_score(self):
        d = {}
        scores = []
        for pred in self.pool_estimators.get_preds():
            estimator = "---".join(pred["estimator"])
            idxs = pred["idxs"]
            y_pred = pred["y_pred"]
            scores += [pred["score"]]
            for idx_, y_ in zip(idxs, y_pred):
                #  TODO handle multioutput
                if self.is_clf:
                    d[(estimator, idx_)] = y_[1]  # classification y_[1]
                else:
                    d[(estimator, idx_)] = y_  #  regression
        cols = sorted(set([k[0] for k, _ in d.items()]))
        cols_idx = dict((c, i) for i, c in enumerate(cols))
        rows = sorted(set([k[1] for k, _ in d.items()]))
        rows_idx = dict((r, i) for i, r in enumerate(rows))
        # do y matrix
        logger.debug("Assembling y")
        # TODO: batch
        with open(
            os.path.join(self.dir, DATA_SUBFOLDER, "batch-0", "y.npy"), "rb"
        ) as f:
            y = np.load(f)
        y = y[rows]
        logger.debug("Assembling X")
        # TODO: Deal with multioutput
        X = np.full((len(rows), len(cols)), np.nan)
        for k, v in d.items():
            i = rows_idx[k[1]]
            j = cols_idx[k[0]]
            X[i, j] = v
        return X, y, np.array(scores)

    def pool(self):
        X, y, scores = self._get_X_y_score()
        mt = MetaTransformer(scores)
        X_t = mt.transform(X)
        mt_file = os.path.join(self.dir, POOL_SUBFOLDER, "meta.pkl")
        joblib.dump(mt, mt_file)
        mdl = MetaModel(self.is_clf)
        mdl.fit(X_t, y)
        mdl_file = os.path.join(self.dir, POOL_SUBFOLDER, "model.pkl")
        joblib.dump(mdl.mdl, mdl_file)
        if self.is_clf:
            y_pred = mdl.mdl.predict_proba(X_t)
        else:
            y_pred = mdl.mdl.predict(X_t)
        metric = Metric(self.is_clf)
        score = metric.score(y, y_pred)
        with open(os.path.join(self.dir, POOL_SUBFOLDER, "eval.json"), "w") as f:
            json.dump(score, f, indent=4)
