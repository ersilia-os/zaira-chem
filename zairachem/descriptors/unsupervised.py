import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from umap import UMAP


MAX_NA = 0.2

class NanFilter(object):
    def __init__(self):
        pass

    def fit(self, X):
        max_na = int((1-MAX_NA)*X.shape[0])
        idxs = []
        for j in X.shape[1]:
            c = np.sum(np.isnan(X[:, j]))
            if c > max_na:
                continue
            else:
                idxs += [j]
        self.col_idxs = idxs

    def transform(self, X):
        return X[:, self.col_idxs]

    def save(self, file_name):
        joblib.dump(self)

    def load(self, file_name):
        return joblib.load(file_name)


class Scaler(object):
    def __init__(self):
        self.abs_limit = 10

    def fit(self, X):
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, X):
        X = self.scaler.transform(X)
        X = np.clip(X, -self.abs_limit, self.abs_limit)
        return X

    def save(self, file_name):
        joblib.dump(self)

    def load(self, file_name):
        return joblib.load(file_name)


class Imputer(object):

    def __init__(self):
        self._fallback = 0

    def fit(self, X):
        ms = []
        for j in range(X.shape[1]):
            vals = X[:, j]
            mask = ~np.isnan(vals)
            vals = vals[mask]
            if len(vals) == 0:
                m = self._fallback
            else:
                m = np.median(vals)
            ms += [m]
        self.impute_values = np.array(ms)

    def transform(self, X):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.impute_values[j]
        return X

    def save(self, file_name):
        joblib.dump(self)

    def load(self, file_name):
        return joblib.load(file_name)


class VarianceFilter(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def save(self, file_name):
        joblib.dump(self)

    def load(self, file_name):
        return joblib.load(file_name)


class Pca(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def save(self, file_name):
        joblib.dump(self)

    def load(self, file_name):
        return joblib.load(file_name)


class OptSne(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass


class UnsupervisedUmap(object):
    def __init__(self):
        pass

    def fit(self, X):
        self.reducer = UMAP(densmap=False)
        self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.transform(X)

    def save(self, file_name):
        joblib.dump(self)

    def load(self, file_name):
        return joblib.load(file_name)


class UnsupervisedTransformations(object):
    def __init__(self):
        pass
