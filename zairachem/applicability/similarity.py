from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import joblib
import numpy as np
import pandas as pd
import faiss
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors
from ..utils.matrices import Hdf5


CPU = 1
faiss.omp_set_num_threads(CPU)

N_NEIGHBORS = 5


class FaissSimilarity(object):
    def __init__(self, n_neighbors=N_NEIGHBORS):
        self.n_neighbors = n_neighbors
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
        m_norm = LA.norm(V, axis=1)
        D, I = self.index.search(V / m_norm[:, None], self.n_neighbors)
        return D, I

    def save(self, path):
        faiss.write_index(self.index, path)

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


class TanimotoSimilarity(object):
    def __init__(self, avoid_self=True, n_neighbors=5):
        self.avoid_self = avoid_self
        self.n_neighbors = n_neighbors

    def _get_mols(self, smiles_list):
        return [Chem.MolFromSmiles(m) for m in smiles_list]

    def _get_fingerprints(self, mols):
        return [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols]

    def _exact_match(self, mol_1, mol_2):
        if mol_1.HasSubstructMatch(mol_2) and mol_2.HasSubstructMatch(mol_1):
            return True
        else:
            return False

    def fit(self, smiles_list):
        self.train_smiles_list = smiles_list
        self.train_mols = self._get_mols(self.train_smiles_list)
        self.train_fps = self._get_fingerprints(self.train_mols)

    def kneighbors(self, smiles_list, as_dataframe=True):
        mols = self._get_mols(smiles_list)
        fps = self._get_fingerprints(mols)
        R = []
        for i in range(len(mols)):
            mol = mols[i]
            fp = fps[i]
            exacts = [
                self._exact_match(mol, train_mol) for train_mol in self.train_mols
            ]
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.train_fps)
            idxs = np.argsort(sims)[::-1]
            idxs = idxs[: self.n_neighbors]
            r = []
            for idx in idxs:
                r += [(self.train_smiles_list[idx], sims[idx], exacts[idx])]
            R += [r]
        if as_dataframe:
            return self.as_flat_dataframe(R)
        return R

    def as_flat_dataframe(self, R):
        S = []
        for r in R:
            smiles = [x[0] for x in r]
            sims = [x[1] for x in r]
            exacts = [x[2] for x in r]
            S += [sims + exacts + smiles]
        columns = (
            ["sim_{0}".format(i) for i in range(self.n_neighbors)]
            + ["exact_{0}".format(i) for i in range(self.n_neighbors)]
            + ["train_smiles_{0}".format(i) for i in range(self.n_neighbors)]
        )
        return pd.DataFrame(S, columns=columns)
