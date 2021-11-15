from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import joblib
from ..automl.flaml import FlamlClassifier

from ..tools.macest.confidence import MacestClassification, MacestRegression


TEST_SIZE = 0.66

ESTIMATORS = ["lgbm"]
TIME_BUDGET = 10


class ConfidenceClassification(object):
    def __init__(self, time_budget=TIME_BUDGET):
        self.time_budget = time_budget

    def fit(self, X, y):
        X_train, X_conf, y_train, y_conf = train_test_split(
            X, y, stratify=y, test_size=TEST_SIZE
        )
        self.mdl = FlamlClassifier()
        self.mdl.fit(
            X_train, y_train, time_budget=self.time_budget, estimators=ESTIMATORS
        )
        self.mac = MacestClassification(self.mdl)
        self.mac.fit(X_conf, y_conf)

    def predict(self, X):
        return self.mdl.predict(X)

    def predict_proba(self, X):
        return self.mdl.predict_proba(X)

    def confidence(self, X):
        return self.mac.confidence(X)

    def save(self, file_name):
        pass
        # joblib.dump(self, file_name)

    def load(self, file_name):
        pass
        # return joblib.load(file_name)
