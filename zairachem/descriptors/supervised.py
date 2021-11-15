from lol import LOL
from umap import UMAP
import joblib


class RfeCv(object):
    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass


class LolliPop(object):
    def __init__(self):
        self._name = "lollipop"

    def fit(self, X, y):
        n_components = min(1024, X.shape[0])
        self.lmao = LOL(n_components=n_components, svd_solver="full")
        self.lmao.fit(X, y)

    def transform(self, X):
        return self.lmao.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class SupervisedUmap(object):
    def __init__(self):
        self._name = "supervised_umap"

    def fit(self, X, y):
        self.reducer = UMAP(densmap=False)
        self.reducer.fit(X, y)

    def transform(self, X):
        return self.reducer.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class SupervisedTransformations(object):
    def __init__(self):
        pass

    def run(self):
        pass
