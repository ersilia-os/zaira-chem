import os
import collections
import pandas as pd
import numpy as np

from lazyqsar.regression.morgan import MorganRegressor
from lazyqsar.binary.morgan import MorganBinaryClassifier


class BaselineEstimator(object):
    def __init__(self, save_path):
        self.save_path = os.path.abspath(save_path)

    def fit(self, data, labels):
        columns = list(data.columns)
        labels_set = set(labels)
        X_columns = [c for c in columns if c not in labels_set]
        X = np.array(data[X_columns], dtype=int)
        
        self.model = BaselineModel(self.save_path)
        self.model.fit(data)

    def save(self):
        pass

    def load(self):
        return BaselineEstimatorArtifact(BaselineModel(self.save_path))


class BaselineEstimatorArtifact(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        df = self.model.predict(data)
        df = df[[c for c in list(df.columns) if c != SMILES_COLUMN]]
        data = collections.OrderedDict()
        for c in list(df.columns):
            if "clf_" in c:
                y_hat = list(df[c])
                data[c] = y_hat
                b_hat = []
                for yh in y_hat:
                    if yh > 0.5:
                        b_hat += [1]
                    else:
                        b_hat += [0]
                data[c + "_bin"] = b_hat
            else:
                data[c] = list(df[c])
        return pd.DataFrame(data)

    def run(self, data):
        results = self.predict(data)
        return results
