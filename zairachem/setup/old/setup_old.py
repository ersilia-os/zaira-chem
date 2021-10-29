import os
import json
import csv
import shutil
import random
import numpy as np
import pandas as pd

from rdkit import Chem
from standardiser import standardise

from .. import logger

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils.multiclass import type_of_target
from sklearn.impute import SimpleImputer


DATA_SUBFOLDER = "data"
DESCRIPTORS_SUBFOLDER = "descriptors"
MODELS_SUBFOLDER = "models"
POOL_SUBFOLDER = "pool"
LITE_SUBFOLDER = "lite"

BATCH_PREFIX = "batch"
SPLIT_PREFIX = "split"
SPLITS_SUBFOLDER = "splits"

MAX_ROWS_PER_BATCH = 100000

N_SPLITS = 5
TEST_SIZE = 0.2

_INPUT_FILENAME = "input.csv"
_CONFIG_FILENAME = "config.json"

_Y_FILENAME = "y.npy"
_TRAIN_IDX_FILENAME = "train_idx.npy"
_TEST_IDX_FILENAME = "test_idx.npy"
_SMILES_FILENAME = "data.smi"

_MAX_BATCHES = 10
_BATCHING_SAMPLING_CHANCE = 0.95


class StandardizeSmiles(object):
    def __init__(self, soft):
        self.soft = soft

    def _smiles(self, mol):
        return Chem.MolToSmiles(mol)

    def _inchikey(self, mol):
        return Chem.inchi.MolToInchiKey(mol)

    def standardize(self, smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        if not self.soft:
            try:
                mol = standardise.run(mol)
            except:
                return None
            if mol is None:
                return None
        ik = self._inchikey(mol)
        smi = self._smiles(mol)
        return ik, smi


class Batcher(object):
    def __init__(self, input_file, output_dir):
        self.input_file = os.path.abspath(input_file)
        self.output_dir = os.path.abspath(output_dir)
        self.max_samples = MAX_ROWS_PER_BATCH
        self.N = self._get_N()

    def _get_N(self):
        with open(self.input_file, "r") as f:
            n = 0
            for l in f:
                n += 1
        return n - 1

    @staticmethod
    def _get_resamp(s, N, chance):
        p = 1 - float(s) / N
        return np.log(1 - chance) / np.log(p)

    def calc_num_batches(self):
        if self.N <= self.max_samples:
            self.samples = self.N
            self.num_batches = 1
        else:
            self.samples = self.max_samples
            self.num_batches = int(
                np.ceil(self._get_resamp(self.smaples, self.N, self.chance))
            )

    def _batches_idxs(self):
        if self.num_batches == 1:
            yield set([i for i in range(self.samples)])
        else:
            splits = ShuffleSplit(n_splits=self.num_batches, train_size=self.samples)
            aux = [i for i in range(self.N)]
            for idxs, _ in splits.split(X=aux, y=aux):
                yield set(idxs)

    def _filter_file_by_batch_idxs(self, idxs):
        lines = []
        with open(self.input_file, "r") as f:
            header = next(f)
            for i, l in enumerate(f):
                if i in idxs:
                    lines += [l]
        random.shuffle(lines)
        return lines, header

    def _get_batch_folder(self, i):
        return os.path.join(
            self.output_dir, DATA_SUBFOLDER, "{0}-{1}".format(BATCH_PREFIX, i)
        )

    def write_batches(self):
        logger.debug("Calculating number of batches")
        self.calc_num_batches()
        logger.debug("Batching")
        for i, idxs in enumerate(self._batches_idxs()):
            lines, header = self._filter_file_by_batch_idxs(idxs)
            batch_folder = self._get_batch_folder(i)
            os.makedirs(batch_folder, exist_ok=True)
            input_file = os.path.join(batch_folder, _INPUT_FILENAME)
            with open(input_file, "w") as f:
                f.write(header)
                for l in lines:
                    f.write(l)


# TODO split Y smartly. For example, take into account the mutual overlap between coverage, in order to provide a split that maximizes coverage.
#  This can be done, in principle, with
class ProcessY(object):
    def __init__(self, y, is_clf):
        self.y_orig = y
        self.is_clf = is_clf
        self.y = None
        self.dtype = self.y_orig.dtype

    def impute(self):
        if self.is_clf:
            strategy = "most_frequent"
        else:
            strategy = "median"
        imp = SimpleImputer(strategy=strategy)
        imp.fit(self.y_orig)
        y = imp.transform(self.y_orig)
        self.y = np.array(y, dtype=self.dtype)

    def process(self):
        self.impute()
        return self.y


class Splitter(object):
    def __init__(self, smiles, y, is_clf):
        self.smiles = smiles
        self.y = y
        self.is_clf = is_clf

    def split(self):
        if self.is_clf:
            splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        else:
            splitter = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        for train_idx, test_idx in splitter.split(self.smiles, self.y):
            yield train_idx, test_idx


class TrainSetup(object):
    def __init__(self, input_file, output_dir, time_budget, standardize):
        self.input_file = os.path.abspath(input_file)
        self.output_dir = os.path.abspath(output_dir)
        self.standard_input_file = os.path.join(
            self.output_dir, DATA_SUBFOLDER, _INPUT_FILENAME
        )
        self.time_budget = time_budget
        self.standardized = standardized

    def _make_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def _make_subfolder(self, name):
        os.makedirs(os.path.join(self.output_dir, name))

    def _standardize_input_file(self):
        standardize_smiles = StandardizeSmiles(soft=not self.standardized)
        self.schema = FileReader(self.input_file).schema()
        logger.debug("Found schema {0}".format(self.schema))
        logger.debug(
            "Reading input file for standardization {0}".format(self.input_file)
        )
        with open(self.input_file, "r") as fi:
            with open(self.standard_input_file, "w") as fo:
                reader = csv.reader(fi)
                header = next(reader)
                smi_idx = header.index(self.schema["smiles"])
                col_idxs = [header.index(col) for col in self.schema["columns"]]
                fo.write(
                    ",".join(
                        [
                            self.schema["inchikey"],
                            self.schema["standard_smiles"],
                            self.schema["smiles"],
                        ]
                        + self.schema["columns"]
                    )
                    + os.linesep
                )
                for r in reader:
                    smi = r[smi_idx]
                    ss = standardize_smiles.standardize(smi)
                    if ss is not None:
                        ik, std_smi = ss
                        r_ = [ik, std_smi, smi] + [r[i] for i in col_idxs]
                        fo.write(",".join(r_) + os.linesep)

    def _batched_input(self):
        logger.debug(
            "Standardizing input file {0} and saving to {1}".format(
                self.input_file, self.standard_input_file
            )
        )
        self._standardize_input_file()
        batcher = Batcher(self.standard_input_file, self.output_dir)
        batcher.write_batches()
        for batch in os.listdir(os.path.join(self.output_dir, DATA_SUBFOLDER)):
            if batch[: len(BATCH_PREFIX)] == BATCH_PREFIX:
                batch_folder = os.path.join(self.output_dir, DATA_SUBFOLDER, batch)
                smiles = []
                y = []
                with open(os.path.join(batch_folder, _INPUT_FILENAME), "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for r in reader:
                        smiles += [r[1]]
                        y += [[np.float(x) for x in r[3:]]]
                os.remove(os.path.join(batch_folder, _INPUT_FILENAME))
                logger.debug("Writing batch smiles")
                with open(os.path.join(batch_folder, _SMILES_FILENAME), "w") as f:
                    for smi in smiles:
                        f.write(smi + os.linesep)
                logger.debug("Processing y")
                y = np.array(y)
                self._y_sample = y[:_SNIFF_SAMPLE_SIZE]
                self._type_of_y()
                self._is_classification()
                py = ProcessY(y, self.is_clf)
                y = py.process()
                logger.debug("Writing y")
                with open(os.path.join(batch_folder, _Y_FILENAME), "wb") as f:
                    np.save(f, y, allow_pickle=False)

    def _type_of_y(self):
        self._y_type = type_of_target(self._y_sample)

    def _is_classification(self):
        if self._y_type == "unknown":
            raise Exception
        if "continuous" in self._y_type:
            self.is_clf = False
        else:
            self.is_clf = True

    def _config_json(self):
        y_type = self._type_of_y()
        config = {
            "time_budget": self.time_budget,
            "type_of_y": self._y_type,
            "is_clf": self.is_clf,
            "schema": self.schema,
        }
        with open(
            os.path.join(self.output_dir, DATA_SUBFOLDER, _CONFIG_FILENAME), "w"
        ) as f:
            json.dump(config, f, indent=4)

    def _splits_per_batch(self, batch):
        logger.debug("Making splits")
        logger.debug("Loading config file")
        with open(
            os.path.join(self.output_dir, DATA_SUBFOLDER, _CONFIG_FILENAME), "r"
        ) as f:
            config = json.load(f)
        logger.debug("Loading y from batch {0}".format(batch))
        with open(
            os.path.join(self.output_dir, DATA_SUBFOLDER, batch, _Y_FILENAME), "rb"
        ) as f:
            y = np.load(f)
        splits_folder = os.path.join(
            self.output_dir, DATA_SUBFOLDER, batch, SPLITS_SUBFOLDER
        )
        if not os.path.exists(splits_folder):
            logger.debug("Making splits folder {0}".format(splits_folder))
            os.mkdir(splits_folder)
        logger.debug("Appying splitter")
        smiles = y  # TODO
        splitter = Splitter(smiles, y, self.is_clf)
        i = 0
        for train_idx, test_idx in splitter.split():
            split_folder = os.path.join(
                splits_folder, "{0}-{1}".format(SPLIT_PREFIX, i)
            )
            if not os.path.exists(split_folder):
                os.mkdir(split_folder)
            with open(os.path.join(split_folder, _TRAIN_IDX_FILENAME), "wb") as f:
                np.save(f, train_idx)
            with open(os.path.join(split_folder, _TEST_IDX_FILENAME), "wb") as f:
                np.save(f, test_idx)
            i += 1

    def _splits(self):
        for batch in os.listdir(os.path.join(self.output_dir, DATA_SUBFOLDER)):
            if batch[: len(BATCH_PREFIX)] == BATCH_PREFIX:
                self._splits_per_batch(batch)

    def setup(self):
        logger.debug("Preparing folder {0}".format(self.output_dir))
        self._make_output_dir()
        self._make_subfolder(DATA_SUBFOLDER)
        self._make_subfolder(DESCRIPTORS_SUBFOLDER)
        self._make_subfolder(MODELS_SUBFOLDER)
        self._make_subfolder(POOL_SUBFOLDER)
        self._make_subfolder(LITE_SUBFOLDER)
        self._batched_input()
        self._config_json()
        self._splits()


Setup = TrainSetup
