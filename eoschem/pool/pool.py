import os
import json
import numpy as np

from .. import logger

from ..setup.setup import DATA_SUBFOLDER, MODELS_SUBFOLDER, POOL_SUBFOLDER

#  TODO: If multiple batches are available, balance selected models per batch
MAX_ESTIMATORS = 100


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
        with open(os.path.join(self.dir, POOL_SUBFOLDER, "estimators.json"), "w") as f:
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
    def __init__(self, X, scores):
        self.X = X
        self.scores = scores

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

    def transform(self):
        funcs = self._funcs()
        X = np.zeros((X.shape[0], len(funcs)))
        for i, func in enumerate(funcs):
            X[:, i] = func()
        return X


class Pool(object):
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)
        self.pool_estimators = PoolEstimators(self.dir)

    def _get_X_y_score(self):
        d = {}
        scores = []
        for pred in self.pool_estimators.get_preds():
            estimator = "---".join(pred["estimator"])
            idxs = pred["idxs"]
            y_pred = pred["y_pred"]
            scores += [pred["score"]]
            for idx_, y_ in zip(idxs, y_pred):
                d[(estimator, idx_)] = y_[1]  # classification y[1]
        cols = sorted(set([k[0] for k, _ in d.items()]))
        cols_idx = dict((c, i) for i, c in enumerate(cols))
        rows = sorted(set([k[1] for k, _ in d.items()]))
        rows_idx = dict((r, i) for i, r in enumerate(rows))
        # do y matrix
        logger.debug("Assembling y")
        # TODO: batch
        with open(
            os.path.join(self.dir, DATA_SUBFOLDER, "batch-01", "y.npy"), "rb"
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
        with open("/Users/mduran/Desktop/X.npy", "wb") as f:
            np.save(f, X)
        with open("/Users/mduran/Desktop/y.npy", "wb") as f:
            np.save(f, y)
        with open("/Users/mduran/Desktop/scores.npy", "wb") as f:
            np.save(f, scores)
