import os
import json
import csv
import shutil
import numpy as np

from .. import logger

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


DATA_SUBFOLDER = "data"
DESCRIPTORS_SUBFOLDER = "descriptors"
MODELS_SUBFOLDER = "models"
POOL_SUBFOLDER = "pool"
LITE_SUBFOLDER = "lite"

MAX_ROWS = 100000

N_SPLITS = 10
TEST_SIZE = 0.2


class TrainSetup(object):
    def __init__(self, input_file, output_dir, time_budget):
        self.input_file = os.path.abspath(input_file)
        self.output_dir = os.path.abspath(output_dir)
        self.time_budget = time_budget

    def _make_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def _make_subfolder(self, name):
        os.makedirs(os.path.join(self.output_dir, name))

    def _read_input_file(self):
        logger.debug("Reading input file {0}".format(self.input_file))
        with open(self.input_file, "r") as f:
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
        self._y_sample = y[:100]

    def _is_classification(self):
        for val in self._y_sample:
            if int(val) != val:
                return False
        return True

    def _config_json(self):
        config = {
            "time_budget": self.time_budget,
            "is_clf": self._is_classification(),
            "header": self.header,
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
