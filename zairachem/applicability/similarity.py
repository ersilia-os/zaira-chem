import joblib
from ..utils.matrices import Hdf5
import numpy as np
import faiss
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors

CPU = 1
faiss.omp_set_num_threads(CPU)

N_NEIGHBORS = 5


class FaissSimilarity(object):
    def __init__(self):
        self.n_neighbors = N_NEIGHBORS
        self.index = None

    def fit(self, file_name):
        hdf5 = Hdf5(file_name)
        V = hdf5.values()
        dim = V.shape[1]
        index = faiss.IndexFlatIP(dim)
        normst = LA.norm(V, axis=1)
        index.add(V / normst[:, None])
        self.index = index
        self.n_neighbors = min(self.n_neighbors, self.index.ntotal)
        D, I = index.search(V / normst[:, None], self.n_neighbors + 1)
        D = D[:, 1:]
        I = I[:, 1:]
        return D

    def kneighbors(self, file_name):
        hdf5 = Hdf5(file_name)
        V = hdf5.values()
        m_norm = np.linalg.norm(V, axis=1)
        D, I = index.search(V / m_norm[:, None], self.n_neighbors)
        return D, I

    def save(self, path):
        faiss.write_index(index, path)

    def load(self, path):
        self.index = faiss.read_index(path)


class NearestSimilarity(object):
    def __init__(self):
        self.n_neighbors = N_NEIGHBORS + 1
        self.metric = "cosine"

    def fit(self, X):
        self.n_neighbors = min(self.n_neighbors, X.shape[0])
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.nn.fit(X)
        D, _ = self.nn.kneighbors(X, return_distance=True)
        D = D[:, 1:]
        self.background = np.sum(D, axis=1)
        print(len(self.background))

    def pvalue(self, X):
        D, _ = self.nn.kneighbors(X, return_distance=True)
        D = D[:, :-1]
        dists = np.sum(D, axis=1)
        n = len(self.background)
        pvalues = []
        for d in dists:
            b = np.sum(self.background >= d)
            pvalues += [b / n]
        return len(pvalues)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)
