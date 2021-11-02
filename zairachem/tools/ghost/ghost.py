import sys
import os
from sklearn import metrics
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

import ghostml


class Ghost(object):
    def __init__(self, mdl):
        self.mdl = mdl
        self.threshold = None

    def _get_class_balance(self, y):
        return np.sum(y) / len(y)

    def get_threshold(self, X_train, y_train):
        train_probs = self.mdl.predict_proba(X_train)[:, 1]
        max_prop = np.max([self._get_class_balance(y_train), 0.6])
        thresholds = np.round(
            np.arange(0.05, max_prop, 0.05), 2
        )  # TODO revise intervals
        threshold = ghostml.optimize_threshold_from_predictions(
            y_train, train_probs, thresholds, ThOpt_metrics="Kappa"
        )
        self.threshold = threshold
        return threshold

    def predict(self, X):
        preds = self.mdl.predict_proba(X)[:, 1]
        bin = []
        for p in preds:
            if p >= self.threshold:
                bin += [1]
            else:
                bin += [0]
        return np.array(bin)
