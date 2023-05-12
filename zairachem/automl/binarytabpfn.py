import numpy as np
from lol import LOL
import random
import collections
from tabpfn import TabPFNClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
import joblib


class TabPFNBinaryClassifier(object):
    def __init__(self, device="cpu", N_ensemble_configurations=4):
        self.device = device
        self.N_ensemble_configurations = N_ensemble_configurations
        self.max_samples = 1000

    def _get_balanced_datasets(self, X, y):
        try:
            smp = SMOTETomek(sampling_strategy="auto")
            X_0, y_0 = smp.fit_resample(X, y)
        except:
            X_0, y_0 = X, y
        try:
            smp = KMeansSMOTE(sampling_strategy="auto")
            X_1, y_1 = smp.fit_resample(X, y)
        except:
            X_1, y_1 = X, y
        try:
            smp = EditedNearestNeighbours(sampling_strategy="auto")
            X_2, y_2 = smp.fit_resample(X, y)
        except:
            X_2, y_2 = X, y
        results = [(X_0, y_0), (X_1, y_1), (X_2, y_2)]
        return results

    def _cap_samples(self, X, y):
        if X.shape[0] <= self.max_samples:
            return [(X, y)]
        idxs = [i for i in range(X.shape[0])]
        R = []
        for _ in range(3):
            smp_idxs = random.sample(idxs, self.max_samples)
            X_, y_ = X[smp_idxs], y[smp_idxs]
            if np.sum(y_) == 0:
                continue
            R += [(X_, y_)]
        return R

    def _get_ensemble(self, X, y):
        R = []
        for X_0, y_0 in self._get_balanced_datasets(X, y):
            for X_1, y_1 in self._cap_samples(X_0, y_0):
                R += [(X_1, y_1)]
        return R

    def fit(self, X, y):
        self.reducer = LOL(n_components=100)
        self.reducer.fit(X, y)
        X = self.reducer.transform(X)
        self.ensemble = self._get_ensemble(X, y)

    def predict_proba(self, X):
        model = TabPFNClassifier(
            device=self.device, N_ensemble_configurations=self.N_ensemble_configurations
        )
        X = self.reducer.transform(X)
        R = []
        for X_tr, y_tr in self.ensemble:
            # print(X_tr.shape, np.sum(y_tr))
            model.fit(X_tr, y_tr)
            R += [model.predict_proba(X)[:, 1]]
            model.remove_models_from_memory()
        R = np.array(R).T
        y_h1 = np.mean(R, axis=1)
        y_h0 = 1 - y_h1
        y_h = np.array([y_h0, y_h1]).T
        return y_h

    def save(self, file_name):
        data = {
            "device": self.device,
            "N_ensemble_configurations": self.N_ensemble_configurations,
            "reducer": self.reducer,
            "ensemble": self.ensemble,
        }
        joblib.dump(data, file_name)

    def load(self, file_name):
        data = joblib.load(file_name)
        model = TabPFNBinaryClassifier(
            device=data["device"],
            N_ensemble_configurations=data["N_ensemble_configurations"],
        )
        model.ensemble = data["ensemble"]
        model.reducer = data["reducer"]
        return TabPFNClassifierArtifact(model, 0.5)


class Binarizer(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def binarize(self, y_hat):
        y_bin = []
        for y in y_hat:
            if y > self.threshold:
                y_bin += [1]
            else:
                y_bin += [0]
        return np.array(y_bin, dtype=np.uint8)


class TabPFNClassifierArtifact(object):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
        if threshold is not None:
            self.binarizer = Binarizer(self.threshold)
        else:
            self.binarizer = None

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        if self.binarizer is not None:
            y_hat = self.predict_proba(X)
            y_bin = self.binarizer.binarize(y_hat)
        else:
            y_bin = self.model.predict(X)
        return y_bin

    def run(self, X, y=None):
        results = collections.OrderedDict()
        results["main"] = {
            "idxs": None,
            "y": y,
            "y_hat": self.predict_proba(X),
            "b_hat": self.predict(X),
        }
        return results
