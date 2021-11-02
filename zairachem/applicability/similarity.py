import joblib
from ..utils.matrices import Hdf5
import numpy as np
import faiss

CPU = 1
faiss.omp_set_num_threads(CPU)

N_NEIGHBORS = 5


class Similarity(object):
    def __init__(self):
        self.n_neighbors = N_NEIGHBORS
        self.index = None

    def fit(self, file_name):
        hdf5 = Hdf5(file_name)
        V = hdf5.values()
        dim = V.shape[0]
        index = faiss.IndexFlatIP(dim)
        m_norm = np.linalg.norm(V, axis=1)
        index.add(V / m_norm[:, None])
        self.index = index
        D, I = index.search(V / m_norm[:, None], self.n_neighbors + 1)
        D = D[:, 1:]
        I = I[:, 1:]
        return D

    def kneighbors(self, file_name):
        k = min(self.n_neighbors, self.index.ntotal)
        hdf5 = Hdf5(file_name)
        V = hdf5.values()
        m_norm = np.linalg.norm(V, axis=1)
        D, I = index.search(V / m_norm[:, None], k)
        return D, I

    def save(self, path):
        faiss.write_index(index, path)

    def load(self, path):
        self.index = faiss.read_index(path)
