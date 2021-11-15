import os
import csv
import requests
import numpy as np
import joblib
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..descriptors.baseline import Fingerprinter
from ..vars import BASE_DIR

DATA_URL = "https://raw.githubusercontent.com/ersilia-os/ersilia/master/ersilia/io/types/examples/compound.tsv"
EXAMPLE_NEIGHBORS_FILE = "example_neighbors.joblib"
SMILES_FILE = "smiles.joblib"
EXAMPLES_BASE_DIR = "examples"

N_NEIGHBORS = 5

os.makedirs(os.path.join(BASE_DIR, EXAMPLES_BASE_DIR), exist_ok=True)


class ExampleNeighbors(object):
    def __init__(self):
        self.n_neighbors = N_NEIGHBORS
        self.index = None
        self.url = DATA_URL
        self.smiles_file = os.path.join(BASE_DIR, EXAMPLES_BASE_DIR, SMILES_FILE)
        self.example_neighbors_file = os.path.join(
            BASE_DIR, EXAMPLES_BASE_DIR, EXAMPLE_NEIGHBORS_FILE
        )
        if not os.path.exists(self.example_neighbors_file):
            self._fit()
        else:
            data = joblib.load(self.example_neighbors_file)
            self.indices = data["indices"]
            self.smiles = data["smiles"]

    def _get_smiles_from_url(self):
        r = requests.get(self.url)
        text = r.iter_lines()
        smiles = []
        for r in text:
            smiles += [r.decode("utf-8").split("\t")[1]]
        return smiles

    def _get_smiles(self):
        if not os.path.exists(self.smiles_file):
            smiles = self._get_smiles_from_url()
            joblib.dump(smiles, self.smiles_file)
        else:
            smiles = joblib.load(self.smiles_file)
        return smiles

    def _get_fps(self):
        fp = Fingerprinter()
        return fp.calculate(self.smiles)

    def _fit(self):
        self.smiles = self._get_smiles()
        V = self._get_fps()
        nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1, metric="jaccard")
        nn.fit(V)
        self.indices = nn.kneighbors(V, return_distance=False)
        data = {"smiles": self.smiles, "indices": self.indices}
        joblib.dump(data, self.example_neighbors_file)

    def get_indices(self):
        return self.indices

    def get_smiles(self):
        return self.smiles


class CompoundSampler(object):
    def __init__(self):
        en = ExampleNeighbors()
        self.smiles = en.get_smiles()
        self.indices = en.get_indices()

    def sample(self, n):
        assert n < self.indices.shape[0]
        seed = random.randint(0, self.indices.shape[0])
        visited_seed = set()
        visited_seed.update([seed])
        visited = set()
        indices = list(self.indices[seed, :])
        visited.update(indices)
        while len(indices) < n:
            available = list(visited.difference(visited_seed))
            if len(available) < 0:
                available = list(
                    set([i for i in self.indices.shape[0]]).difference(visited_seed)
                )
            seed = random.choice(available)
            indices_ = list(self.indices[seed, :])
            for i in indices_:
                if i not in visited:
                    indices += [i]
                    visited.update([i])
            visited_seed.update([seed])
        return [self.smiles[i] for i in indices[:n]]


class ClassificationMaker(object):
    def __init__(self):
        pass

    def make(self, n, prop_1):
        smiles = CompoundSampler().sample(n)
        n_1 = int(prop_1 * n)
        y = [1] * n_1 + [0] * (n - n_1)
        idxs = [i for i in range(len(y))]
        random.shuffle(idxs)
        return [smiles[i] for i in idxs], [y[i] for i in idxs]


class RegressionMaker(object):
    def __init__(self):
        pass

    def make(self, n):
        smiles = CompoundSampler().sample(n)
        y = np.random.normal(size=len(smiles))
        idxs = np.argsort(y)
        smiles = [smiles[i] for i in idxs]
        y = [y[i] for i in idxs]
        idxs = [i for i in range(len(y))]
        random.shuffle(idxs)
        return [smiles[i] for i in idxs], [y[i] for i in idxs]
