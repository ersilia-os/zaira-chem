import joblib
import json
import os
import numpy as np
import tempfile

from ..pool.pool import (
    _ESTIMATORS_FILENAME,
    _META_TRANSFORMER_FILENAME,
    _META_MODEL_FILENAME,
)

BATCH_SIZE = 100000


class Predictor(object):
    def __init__(self, input_file, output_file, dir):
        self.dir = os.path.abspath(dir)
        self.input_file = os.path.abspath(input_file)
        self.output_file = os.path.abspath(output_file)
        # self.tmp_folder = tempfile.mkdtemp()
        self.tmp_folder = os.path.abspath("tmp_predict")  # Make it temporary
        self.estimators = self._get_estimators()
        self.mt = self._get_meta_transformer()
        self.mmdl = self._get_meta_model()
        self.descriptor_names = self._get_needed_descriptor_names()

    def _get_estimators(self):
        estimators_file = os.path.join(self.dir, POOL_SUBFOLDER, _ESTIMATORS_FILENAME)
        with open(estimators_file, "r") as f:
            estimators = json.load(f)
        return estimators

    def _get_meta_transformer(self):
        mt_file = os.path.join(self.dir, POOL_SUBFOLDER, _META_TRANSFORMER_FILENAME)
        return joblib.load(mt_file)

    def _get_meta_model(self):
        mdl_file = os.path.join(self.dir, POOL_SUBFOLDER, _META_MODEL_FILENAME)
        return joblib.load(mdl_file)

    def _get_needed_descriptor_names(self):
        descs = set()
        for estim in self.estimators:
            descs.update([estim[0][1]])
        return descs

    @staticmethod
    def _batch_iterator(smiles):
        l = smiles
        n = BATCH_SIZE
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def setup(self):
        logger.debug(
            "Setting up predictor. Making descriptor folders in {0}".format(
                self.tmp_folder
            )
        )
        for descriptor_name in self.descriptor_names:
            dir_ = os.path.join(self.tmp_folder, descriptor_name)
            os.makedirs(dir_, exist_ok=True)
        logger.debug()

    def predict(self):
        logger.debug("Getting descriptors")

        pass


class PredictorSetup(object):
    def __init__(self, dir, output_dir):
        self.dir = os.path.abspath(dir)
        self.output_dir = os.path.abspath(output_dir)


class Predictor(object):
    def __init__(self, smiles, models_path):
        self.smiles = smiles
        self.models_path = os.path.abspath(models_path)
        self.max_n = MAX_N

    def _search_models(self):
        for model_dir in os.listdir(self.models_path):
            if model_dir[:5] != "model":
                continue
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
        sel_models = all_models[
            : self.max_n
        ]  # TO-DO select based on performance / for the moment it is random.
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
            yield l[i : i + n]

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
                        p = mdl.predict_proba(X)[:, 1]
                    else:
                        p = mdl.predict(X)
                P += [p]
        P = np.array(P).T
        return np.mean(P, axis=1)
