import os
import json
import csv
import shutil
import numpy as np
import pandas as pd

from rdkit import Chem

from .. import logger

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.utils.multiclass import type_of_target


DATA_SUBFOLDER = "data"
DESCRIPTORS_SUBFOLDER = "descriptors"
MODELS_SUBFOLDER = "models"
POOL_SUBFOLDER = "pool"
LITE_SUBFOLDER = "lite"

MAX_ROWS = 100000

N_SPLITS = 10
TEST_SIZE = 0.2

_SNIFF_SAMPLE_SIZE = 1000
_MIN_CORRECT_SMILES = 0.8


class FileReader(object):

    def __init__(self, input_file):
        self.input_file = os.path.abspath(input_file)
        self.df_ = pd.read_csv(self.input_file, nrows=_SNIFF_SAMPLE_SIZE)
        self.columns = list(self.df_.columns)

    def _prop_correct_smiles(self, col):
        values = list(self.df_[col])
        c = 0
        for v in values:
            try:
                mol = Chem.MolFromSmiles(v)
            except:
                continue
            if mol is not None:
                c += 1
        return float(c)/len(values)

    def _is_data_column(self, col):
        values = list(self.df_[self.df_[col].notnull()][col])
        c = 0
        for v in values:
            try:
                float(v)
            except:
                continue
            c += 1
        if c == len(values):
            return True
        else:
            return False

    def _find_smiles_column(self):
        cands = []
        for col in self.columns:
            if self._prop_correct_smiles(col) > _MIN_CORRECT_SMILES:
                cands += [col]
            else:
                continue
        if len(cands) != 1:
            raise Exception
        else:
            return cands[0]

    def _find_data_columns(self):
        datacols = []
        for col in self.columns:
            if self._is_data_column(col):
                datacols += [col]
            else:
                continue
        return datacols

    def schema(self):
        logger.debug("Guessing schema")
        d = {
            "smiles": self._find_smiles_column(),
            "columns": self._find_data_columns()
        }
        logger.debug("Guessed schema {0}".format(d))
        return d


class TrainSetup(object):
    def __init__(self, input_file, output_dir, time_budget):
        self.input_file = os.path.abspath(input_file)
        self.output_dir = os.path.abspath(output_dir)
        self.standard_input_file = os.path.join(self.output_dir, DATA_SUBFOLDER, "input.csv")
        self.time_budget = time_budget

    def _make_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def _make_subfolder(self, name):
        os.makedirs(os.path.join(self.output_dir, name))

    @staticmethod
    def _is_valid_smiles(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            return False
        if mol is None:
            return False
        else:
            return True

    def _standardize_input_file(self):
        self.schema = FileReader(self.input_file).schema()
        logger.debug("Reading input file for standardization {0}".format(self.input_file))
        with open(self.input_file, "r") as fi:
            with open(self.standard_input_file, "w") as fo:
                reader = csv.reader(fi)
                header = next(reader)
                smi_idx = header.index(self.schema["smiles"])
                col_idxs = [header.index(col) for col in self.schema["columns"]]
                fo.write(",".join([self.schema["smiles"]]+self.schema["columns"])+os.linesep)
                for r in reader:
                    smi = r[smi_idx]
                    if self._is_valid_smiles(smi):
                        r_ = [r[smi_idx]] + [r[i] for i in col_idxs]
                        fo.write(",".join(r_)+os.linesep)

    def _read_input_file(self):
        logger.debug("Standardizing input file first")
        self._standardize_input_file()
        logger.debug("Reading standard input file {0}".format(self.standard_input_file))
        with open(self.standard_input_file, "r") as f:
            reader = csv.reader(f)
            self.header = next(reader)
            for r in reader:
                yield r[0], [float(r_) for r_ in r[1:]]

    def _batched_input(self):
        smiles = []
        y = []
        for smi, y_ in self._read_input_file():
            smiles += [smi]
            y += [y_]
        #  TODO iterate over batches
        batch_folder = os.path.join(self.output_dir, DATA_SUBFOLDER, "batch-01")
        os.mkdir(batch_folder)
        with open(os.path.join(batch_folder, "data.smi"), "w") as f:
            for smi in smiles:
                f.write("{0}{1}".format(smi, os.linesep))
        with open(os.path.join(batch_folder, "y.npy"), "wb") as f:
            y = np.array(y)
            np.save(f, y, allow_pickle=False)
        self._y_sample = y[:_SNIFF_SAMPLE_SIZE]

    def _type_of_y(self):
        return type_of_target(self._y_sample)

    @staticmethod
    def _is_classification(y_type):
        if y_type == "unknown":
            raise Exception
        if "continuous" in y_type:
            return False
        else:
            return True

    def _config_json(self):
        y_type = self._type_of_y()
        config = {
            "time_budget": self.time_budget,
            "type_of_y": y_type,
            "is_clf": self._is_classification(y_type),
            "schema": self.schema,
        }
        with open(
            os.path.join(self.output_dir, DATA_SUBFOLDER, "config.json"), "w"
        ) as f:
            json.dump(config, f, indent=4)

    def _splits_per_batch(self, batch):
        logger.debug("Making splits")
        logger.debug("Loading config file")
        with open(
            os.path.join(self.output_dir, DATA_SUBFOLDER, "config.json"), "r"
        ) as f:
            config = json.load(f)
        logger.debug("Loading y from batch {0}".format(batch))
        with open(
            os.path.join(self.output_dir, DATA_SUBFOLDER, batch, "y.npy"), "rb"
        ) as f:
            y = np.load(f)
        splits_folder = os.path.join(self.output_dir, DATA_SUBFOLDER, batch, "splits")
        if not os.path.exists(splits_folder):
            logger.debug("Making splits folder {0}".format(splits_folder))
            os.mkdir(splits_folder)
        if config["is_clf"]:
            splitter = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        else:
            splitter = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_SIZE)
        i = 0
        for train_idx, test_idx in splitter.split(y, y):
            split_folder = os.path.join(splits_folder, "split-{0}".format(i))
            if not os.path.exists(split_folder):
                os.mkdir(split_folder)
            with open(os.path.join(split_folder, "train_idx.npy"), "wb") as f:
                np.save(f, train_idx)
            with open(os.path.join(split_folder, "test_idx.npy"), "wb") as f:
                np.save(f, test_idx)
            i += 1

    def _splits(self):
        for batch in os.listdir(os.path.join(self.output_dir, DATA_SUBFOLDER)):
            if batch[:5] == "batch":
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
