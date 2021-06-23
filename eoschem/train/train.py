import os
import csv
import numpy as np
import joblib
import shutil
import json

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from ..descriptors.descriptors import DescriptorsCalculator
from .finder import ModelFinder

MAX_N = 70000
MAX_FOLDS = 3

class Trainer(object):

    def __init__(self, input_file, output_path):
        self.input_file = os.path.abspath(input_file)
        self.output_path = os.path.abspath(output_path)
        self._make_output_path()
        self._read_input()
        self.is_clf = self._is_classification()
        self.finder = ModelFinder(self.is_clf)

    def _make_output_path(self):
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        os.makedirs(self.output_path)

    def _read_input_lines(self):
        with open(self.input_file, "r") as f:
            reader = csv.reader(f)
            self.header = next(reader)
            for r in reader:
                yield r[0], float(r[1])

    def _read_input(self):
        self.smiles = []
        self.y = []
        for smi, value in self._read_input_lines():
            self.smiles += [smi]
            self.y += [value]
        self.y = np.array(self.y)

    def _is_classification(self):
        for val in self.y:
            if int(val) != val:
                return False
        return True

    def _get_fitted_model(self, X, y, mdl=None):
        if mdl is None:
            mdl, params = self.finder.find_model(X, y)
        else:
            params = None
        mdl.fit(X, y)
        return mdl, params

    def _estimate_folds(self):
        if len(self.y) <= MAX_N:
            return None, None
        n_folds = int(np.min([np.ceil(len(self.y)/MAX_N), MAX_FOLDS]))
        train_size = np.min([MAX_N, len(self.y)])
        return n_folds, train_size

    def _batch_iterator(self):
        n_folds, train_size = self._estimate_folds()
        if n_folds is None:
            yield 0, self.smiles, self.y
        else:
            if self.is_clf:
                spl = StratifiedShuffleSplit(n_splits=n_folds, train_size=train_size)
            else:
                spl = ShuffleSplit(n_splits=n_folds, train_size=train_size)
            i = -1
            for idxs, _ in spl.split(X=self.smiles, y=self.y):
                i += 1
                smiles = [self.smiles[i] for i in idxs]
                y = self.y[idxs]
                yield i, smiles, y

    def train(self):
        mdl = None
        for batch, smiles, y in self._batch_iterator():
            descriptors = DescriptorsCalculator(smiles)
            for i, d in enumerate(descriptors.calculate()):
                X, n = d
                mdl, params = self._get_fitted_model(X, y, mdl=mdl)
                meta = {
                    "batch": batch,
                    "is_clf": self.is_clf,
                    "descriptor": n,
                    "dim": X.shape[1],
                    "output": self.header[1],
                    "params": params
                }
                dest = os.path.join(self.output_path, "model_{0}".format(i))
                if not os.path.exists(dest):
                    os.makedirs(dest)
                    with open(os.path.join(dest, "meta.json"), "w") as f:
                        json.dump(meta, f, indent=4)
                joblib.dump(mdl, os.path.join(dest, "model_{0}.pkl".format(batch)))
