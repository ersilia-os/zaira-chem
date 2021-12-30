import os
import shutil
import json

from time import time

from .. import ZairaBase

from .files import SingleFileForPrediction, ParametersFile
from .tasks import SingleTasksForPrediction
from .standardize import Standardize
from .merge import DataMergerForPrediction
from .clean import SetupCleaner

from . import PARAMETERS_FILE

from ..vars import BASE_DIR
from ..vars import SESSION_FILE

from ..vars import DATA_SUBFOLDER
from ..vars import DESCRIPTORS_SUBFOLDER
from ..vars import MODELS_SUBFOLDER
from ..vars import POOL_SUBFOLDER
from ..vars import LITE_SUBFOLDER
from ..vars import REPORT_SUBFOLDER
from ..vars import TRAINED_MODEL_SUBFOLDER

from ..tools.melloddy.pipeline import MelloddyTunerPredictPipeline


class PredictSetup(object):
    def __init__(self, input_file, output_dir, model_dir, time_budget):
        self.input_file = os.path.abspath(input_file)
        if output_dir is None:
            self.output_dir = os.path.abspath(self.input_file.split(".")[0])
        else:
            self.output_dir = os.path.abspath(output_dir)
        assert not os.path.exists(self.output_dir), "Output dir exists!"
        assert model_dir is not None, "Model directory not specified"
        self.model_dir = os.path.abspath(model_dir)
        self.time_budget = time_budget  # TODO
        assert os.path.exists(self.model_dir)

    def _open_session(self):
        data = {
            "output_dir": self.output_dir,
            "model_dir": self.model_dir,
            "time_stamp": int(time()),
            "elapsed_time": 0,
        }
        with open(os.path.join(BASE_DIR, SESSION_FILE), "w") as f:
            json.dump(data, f, indent=4)

    def _make_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def _make_subfolder(self, name):
        os.makedirs(os.path.join(self.output_dir, name))

    def _make_subfolders(self):
        self._make_subfolder(DATA_SUBFOLDER)
        self._make_subfolder(DESCRIPTORS_SUBFOLDER)
        self._make_subfolder(MODELS_SUBFOLDER)
        self._make_subfolder(POOL_SUBFOLDER)
        self._make_subfolder(LITE_SUBFOLDER)
        self._make_subfolder(REPORT_SUBFOLDER)
        os.symlink(
            self.model_dir, os.path.join(self.output_dir, TRAINED_MODEL_SUBFOLDER)
        )
        shutil.copyfile(
            os.path.join(self.model_dir, DATA_SUBFOLDER, PARAMETERS_FILE),
            os.path.join(self.output_dir, DATA_SUBFOLDER, PARAMETERS_FILE),
        )

    def _normalize_input(self):
        params = ParametersFile(
            full_path=os.path.join(self.model_dir, DATA_SUBFOLDER, PARAMETERS_FILE)
        ).load()
        f = SingleFileForPrediction(self.input_file, params)
        f.process()
        self.has_tasks = f.has_tasks

    def _melloddy_tuner_run(self):
        MelloddyTunerPredictPipeline(os.path.join(self.output_dir, DATA_SUBFOLDER)).run(
            has_tasks=self.has_tasks
        )

    def _standardize(self):
        Standardize(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()

    def _tasks(self):
        SingleTasksForPrediction(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()

    def _merge(self):
        DataMergerForPrediction(os.path.join(self.output_dir, DATA_SUBFOLDER)).run(
            self.has_tasks
        )

    def _clean(self):
        SetupCleaner(os.path.join(self.output_dir, DATA_SUBFOLDER)).run()

    def update_elapsed_time(self):
        ZairaBase().update_elapsed_time()

    def setup(self):
        self._make_output_dir()
        self._open_session()
        self._make_subfolders()
        self._normalize_input()
        self._melloddy_tuner_run()
        self._standardize()
        if self.has_tasks:
            self._tasks()
        self._merge()
        self._clean()
        self.update_elapsed_time()
