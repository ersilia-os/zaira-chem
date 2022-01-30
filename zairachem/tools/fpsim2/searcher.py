import os
import csv
import random
from FPSim2.io import create_db_file
from FPSim2 import FPSim2Engine

root = os.path.dirname(os.path.abspath(__file__))


class SimilaritySearcher(object):
    def __init__(self, fp_filename=None, n_workers=4):
        if fp_filename is None:
            self.fp_filename = self.reference_file()
        else:
            self.fp_filename = fp_filename
        self.db_smiles_filename = self.fp_filename[:-3] + ".csv"
        if os.path.exists(self.fp_filename):
            self.engine = FPSim2Engine(self.fp_filename)
            self.db_smiles = self.read_db_smiles()
        else:
            self.engine = None
            self.db_smiles = None
        self.n_workers = n_workers

    def reference_file(self):
        return os.path.join(root, "data", "reference_library.h5")

    def read_db_smiles(self):
        smiles_list = []
        with open(self.db_smiles_filename, "r") as f:
            reader = csv.reader(f)
            for r in reader:
                smiles_list += [r[0]]
        return smiles_list

    def write_db_smiles(self, smiles_list):
        with open(self.db_smiles_filename, "w") as f:
            writer = csv.writer(f)
            for smi in smiles_list:
                writer.writerow([smi])

    def fit(self, smiles_list):
        self.write_db_smiles(smiles_list)
        smiles_list = [[smi, i] for i, smi in enumerate(smiles_list)]
        create_db_file(
            smiles_list, self.fp_filename, "Morgan", {"radius": 2, "nBits": 2048}
        )

    def search(self, smiles, cutoff=0.7):
        results = self.engine.similarity(smiles, cutoff, n_workers=self.n_workers)
        results = [(r[0], self.db_smiles[r[0]], r[1]) for r in results]
        return results


class RandomSearcher(object):
    def __init__(self, db_smiles_filename=None):
        if db_smiles_filename is None:
            self.db_smiles_filename = self.reference_smiles_file()
        else:
            self.db_smiles_filename = db_smiles_filename
        self.db_smiles = self.read_db_smiles()

    def reference_smiles_file(self):
        return os.path.join(root, "data", "reference_library.csv")

    def read_db_smiles(self):
        smiles_list = []
        with open(self.db_smiles_filename, "r") as f:
            reader = csv.reader(f)
            for r in reader:
                smiles_list += [r[0]]
        return smiles_list

    def search(self, n):
        return set(random.sample(self.db_smiles, n))
