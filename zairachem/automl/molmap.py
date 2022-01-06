import os
import collections
import pandas as pd

from zairachem.setup import SMILES_COLUMN

from ..tools.molmap.molmap import MolMapModel


class MolMapEstimator(object):
    def __init__(self, save_path):
        self.save_path = os.path.abspath(save_path)

    def fit(self, data, labels):
        data = data[[SMILES_COLUMN] + labels]
        self.model = MolMapModel(self.save_path)
        self.model.fit(data)

    def save(self):
        pass

    def load(self):
        return MolMapEstimatorArtifact(MolMapModel(self.save_path))


class MolMapEstimatorArtifact(object):
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
