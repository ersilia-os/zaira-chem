import h5py
import faiss
import csv
import requests
import numpy as np
from sklearn.neighbors import NearestNeighbors


N_NEIGHBORS = 5

# TODO: A longer list of smiles
DATA_URL = "https://raw.githubusercontent.com/ersilia-os/ersilia/master/ersilia/io/types/examples/compound.tsv"


class ExampleNeighbors(object):

    def __init__(self):
        self.n_neighbors = N_NEIGHBORS
        self.index = None
        self.url = DATA_URL

    def _get_smiles(self):
        r = requests.get(self.url)
        text = r.iter_lines()
        reader = csv.reader(text)
        smiles += []
        for r in reader:
            smiles += [r[1]]
        return smiles

    def _get_fps(self):
        V = []

    def _fit(self):
        self.smiles = self._get_smiles()
        self.V = self._get_fps()
        nn = NearestNeighbors(k_neighbors = self.n_neighbors+1)
        nn.fit(V)
        indices = nn.kneighbors(V, return_distance=False)
        


    def get_indices(self):

    def get_smiles(self):




class CompoundSampler(object):

    def __init__(self, n):



class NetworkBuilder(object):

    def __init__(self, n):





class ClassificationMaker(object):

    def __init__(self):
        pass

    def make(self):
        pass


class RegressionMaker(object):

    def __init__(self):
        pass

    def make(self):
        pass
