import os
import csv
import requests
from rdkit import Chem
from tdc.single_pred import Tox
from standardiser import standardise


# Open Source Malaria Project
OSM_URL = "https://raw.githubusercontent.com/ersilia-os/osm-series4-candidates-2/main/data/raw/series4_processed.csv"


class OsmExample(object):
    def __init__(self):
        self.url = OSM_URL
        self._get_from_url()

    def _get_from_url(self):
        r = requests.get(self.url)
        text = r.iter_lines()
        self.smiles = []
        self.reg = []
        self.clf = []
        for r in text:
            l = r.decode("utf-8").split(",")
            mol = Chem.MolFromSmiles(l[1])
            if mol is None:
                continue
            self.smiles += [l[1]]
            self.reg += [float(l[3])]
            self.clf += [int(float(l[4]))]

    def classification(self, file_name):
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "activity"])
            for i in range(len(self.smiles)):
                writer.writerow([self.smiles[i], self.clf[i]])

    def regression(self, file_name):
        with open(file_name, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "activity"])
            for i in range(len(self.smiles)):
                writer.writerow([self.smiles[i], self.reg[i]])


class ClinToxExample(object):
    def __init__(self):
        self.data = Tox(name="ClinTox")

    def get_splits(self, folder_name):
        os.makedirs(folder_name, exist_ok=True)

        train_file = os.path.join(folder_name, "train.csv")
        test_file = os.path.join(folder_name, "test.csv")

        data = self.data.get_split()

        train = data["train"]
        test = data["test"]

        smiles_train = train["Drug"]
        y_train = train["Y"]

        smiles_tr = []
        y_tr = []
        for smi, y in zip(smiles_train, y_train):
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
            try:
                mol = standardise.run(mol)
            except:
                continue
            if not mol:
                continue
            smiles_tr += [Chem.MolToSmiles(mol)]
            y_tr += [y]

        smiles_test = test["Drug"]
        y_test = test["Y"]

        smiles_te = []
        y_te = []
        for smi, y in zip(smiles_test, y_test):
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
            try:
                mol = standardise.run(mol)
            except:
                continue
            if not mol:
                continue
            smiles_te += [Chem.MolToSmiles(mol)]
            y_te += [y]

        with open(train_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "clintox"])
            for smi, y in zip(smiles_tr, y_tr):
                writer.writerow([smi, y])

        with open(test_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "clintox"])
            for smi, y in zip(smiles_te, y_te):
                writer.writerow([smi, y])
