import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from macest.classification import models as clf_mac
from macest.regression import models as reg_mac


TEST_SIZE = 0.5


class MacestClassification(object):
    def __init__(self, mdl):
        self.mdl = mdl

    def fit(self, X, y):
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=TEST_SIZE)
        self.mac = clf_mac.ModelWithConfidence(self.mdl, X_train, y_train)
        self.mac.fit(X_cal, y_cal)

    def confidence(self, X):
        return self.mac.predict_confidence_of_point_prediction(X)


class MacestRegression(object):
    def __init__(self, mdl):
        self.mdl = mdl

    def fit(self, X, y):
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=TEST_SIZE)
        preds = self.mdl.predict(X_train)
        test_error = abs(preds - y_train)
        self.mac = reg_mac.ModelWithPredictionInterval(self.mdl, X_train, test_error)
        self.mac.fit(X_cal, y_cal)

    def confidence(self, X, conf_level=90):
        self.mac.predict_interval(X, conf_level=conf_level)
