import os
import numpy as np
import json
import joblib
import shutil

from flaml import AutoML

from ..setup.setup import (
    DATA_SUBFOLDER,
    DESCRIPTORS_SUBFOLDER,
    MODELS_SUBFOLDER,
    _CONFIG_FILENAME,
    SPLITS_SUBFOLDER,
    _TRAIN_IDX_FILENAME,
    _TEST_IDX_FILENAME
)

from .. import logger

from ..metrics.metrics import Metric


TIME_BUDGET = 10

_

_


class Fit(object):
    def __init__(self, dir):
        self.dir = os.path.abspath(dir)

    def _is_clf(self):
        data_json = os.path.join(self.dir, DATA_SUBFOLDER, _CONFIG_FILENAME)
        logger.debug("Reading task and metric from {0}".format(data_json))
        with open(data_json, "r") as f:
            data = json.load(f)
        return data["is_clf"]

    def _task_and_metric(self):
        is_clf = self._is_clf()
        if is_clf:
            return ("classification", "auto")
        else:
            return ("regression", "auto")

    def _load_X_y(self, batch, descriptor):
        X_file = os.path.join(
            self.dir, DESCRIPTORS_SUBFOLDER, batch, descriptor, "X.npy"
        )
        logger.debug("Loading X from {0}".format(X_file))
        with open(X_file, "rb") as f:
            X = np.load(f)
        nan_count = np.sum(np.isnan(X))
        if nan_count != 0:
            logger.warning("X contains {0} NaNs.".format(nan_count))
        y_file = os.path.join(self.dir, DATA_SUBFOLDER, batch, "y.npy")
        logger.debug("Loading y from {0}".format(y_file))
        with open(y_file, "rb") as f:
            y = np.load(f)
        nan_count = np.sum(np.isnan(y))
        if nan_count != 0:
            logger.warning("y contains {0} NaNs.".format(nan_count))
        return X, y

    def _find_data(self):
        for batch in os.listdir(os.path.join(self.dir, DESCRIPTORS_SUBFOLDER)):
            if batch[:5] == "batch":
                for descriptor in os.listdir(
                    os.path.join(self.dir, DESCRIPTORS_SUBFOLDER, batch)
                ):
                    if os.path.exists(
                        os.path.join(
                            self.dir, DESCRIPTORS_SUBFOLDER, batch, descriptor, "X.npy"
                        )
                    ):
                        yield (batch, descriptor)

    def _fit_main(self, batch, descriptor):
        X, y = self._load_X_y(batch, descriptor)
        logger.debug("X shape ({0}, {1})".format(X.shape[0], X.shape[1]))
        logger.debug("y shape ({0}, {1})".format(y.shape[0], y.shape[1]))
        # TODO: Work in multioutput scenarios natively
        assert y.shape[1] == 1
        y = y.ravel()
        task, metric = self._task_and_metric()
        logger.info("Task: {0}".format(task))
        logger.info("Metric: {0}".format(metric))
        automl_settings = {
            "time_budget": int(TIME_BUDGET) * 60,  #  in seconds
            "metric": metric,
            "task": task,
            "log_file_name": "automl.log",
            "verbose": 0
        }
        automl = AutoML()
        automl.fit(X_train=X, y_train=y, **automl_settings)
        meta = {
            "batch": batch,
            "descriptor": descriptor,
            "estimator": automl.best_estimator,
            "loss": automl.best_loss,
            "metric": metric,
            "task": task,
        }
        mdl_dir = os.path.join(self.dir, MODELS_SUBFOLDER, batch, descriptor)
        os.makedirs(mdl_dir, exist_ok=True)
        logger.debug("Saving model to {0}".format(mdl_dir))
        mdl = automl.model
        with open(os.path.join(mdl_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
        joblib.dump(mdl, os.path.join(mdl_dir, "model.pkl"))
        logger.debug("Trying to refit")
        mdl.fit(X, y)
        # remove catboost info folder if generated
        cwd = os.getcwd()
        catboost_info = os.path.join(cwd, "catboost_info")
        if os.path.exists(catboost_info):
            shutil.rmtree(catboost_info)
        if os.path.exists("automl.log"):
            os.remove("automl.log")

    def _fit_splits(self, batch, descriptor):
        logger.debug("Fitting splits")
        is_clf = self._is_clf()
        metric = Metric(is_clf)
        mdl_dir = os.path.join(self.dir, MODELS_SUBFOLDER, batch, descriptor)
        mdl = joblib.load(os.path.join(mdl_dir, "model.pkl"))
        X, y = self._load_X_y(batch, descriptor)
        splits_folder = os.path.join(self.dir, DATA_SUBFOLDER, batch, SPLITS_SUBFOLDER)
        for split in os.listdir(splits_folder):
            if split[:len(_SPLIT_PREFIX)] == _SPLIT_PREFIX:
                logger.debug("Training on split {0}".format(split))
                split_folder = os.path.join(splits_folder, split)
                with open(os.path.join(split_folder, _TRAIN_IDX_FILENAME), "rb") as f:
                    train_idx = np.load(f)
                with open(os.path.join(split_folder, _TEST_IDX_FILENAME), "rb") as f:
                    test_idx = np.load(f)
                mdl.fit(X[train_idx], y[train_idx])
                if is_clf:
                    yp = mdl.predict_proba(X[test_idx])
                else:
                    yp = mdl.predict(X[test_idx])
                yt = y[test_idx]
                dir_ = os.path.join(mdl_dir, split)
                if not os.path.exists(dir_):
                    os.mkdir(dir_)
                logger.debug("Saving model to {0}".format(dir_))
                joblib.dump(mdl, os.path.join(dir_, "model.pkl"))
                logger.debug("Evaluating")
                eval_file = os.path.join(dir_, "eval.json")
                eval = metric.score(yt, yp)
                with open(eval_file, "w") as f:
                    json.dump(eval, f, indent=4)
                logger.debug("Saving predictions")
                y_pred_file = os.path.join(dir_, "y_pred.npy")
                with open(y_pred_file, "wb") as f:
                    np.save(f, yp, allow_pickle=False)

    def _fit(self, batch, descriptor):
        self._fit_main(batch, descriptor)
        self._fit_splits(batch, descriptor)

    def fit(self):
        for batch, descriptor in self._find_data():
            self._fit(batch, descriptor)
