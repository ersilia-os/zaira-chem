import csv
import requests
from rdkit import Chem


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
