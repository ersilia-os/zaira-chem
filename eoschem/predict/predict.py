import joblib
import json
import os
import numpy as np

from ..descriptors.descriptors import DescriptorsCalculator

MAX_N = 100
BATCH_SIZE = 100000


class Predictor(object):

    def __init__(self, smiles, models_path):
        self.smiles = smiles
        self.models_path = os.path.abspath(models_path)
        self.max_n = MAX_N

    def _search_models(self):
        for model_dir in os.listdir(self.models_path):
            if model_dir[:5] != "model": continue
            meta_file = os.path.join(self.models_path, model_dir, "meta.json")
            with open(meta_file, "r") as f:
                meta = json.load(f)
            for model_file in os.listdir(os.path.join(self.models_path, model_dir)):
                if model_file[:5] != "model" or model_file[-4:] != ".pkl":
                    continue
                else:
                    model_file = os.path.join(self.models_path, model_dir, model_file)
                    yield meta, model_file

    def _load_model(self, model_file):
        mdl = joblib.load(model_file)
        return mdl

    def _sample_models(self):
        all_models = [(meta, model_file) for meta, model_file in self._search_models()]
        sel_models = all_models[:self.max_n] # TO-DO select based on performance / for the moment it is random.
        return sel_models

    def _get_descriptor_names(self, meta_models):
        names = set()
        for meta, _ in meta_models:
            names.update([meta["descriptor"]])
        return sorted(names)

    def _load_iterator(self, meta_models, descriptor_name):
        for meta, model_file in meta_models:
            if meta["descriptor"] != descriptor_name:
                continue
            mdl = self._load_model(model_file)
            is_clf = meta["is_clf"]
            yield is_clf, mdl

    def _batch_iterator(self):
        l = self.smiles
        n = BATCH_SIZE
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def predict(self):
        models = self._sample_models()
        descriptor_names = self._get_descriptor_names(models)
        P = []
        for n in descriptor_names:
            for smiles_batch in self._batch_iterator():
                descriptors = DescriptorsCalculator(smiles_batch)
                X = descriptors.calculate_one(n)
                for is_clf, mdl in self._load_iterator(models, n):
                    if is_clf:
                        p = mdl.predict_proba(X)[:,1]
                    else:
                        p = mdl.predict(X)
                P += [p]
        P = np.array(P).T
        return np.mean(P, axis=1)
