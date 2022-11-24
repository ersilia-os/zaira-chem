import os
import shutil
import pandas as pd

from ..vars import (
    DESCRIPTORS_SUBFOLDER,
    POOL_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
    REPORT_SUBFOLDER,
    OUTPUT_FILENAME,
    OUTPUT_TABLE_FILENAME,
    PERFORMANCE_TABLE_FILENAME,
    OUTPUT_XLSX_FILENAME,
)
from .. import ZairaBase
from ..estimators import RESULTS_MAPPED_FILENAME


class Cleaner(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)

    def _clean_descriptors_by_subfolder(self, path, subfolder):
        path = os.path.join(path, subfolder)
        for d in os.listdir(path):
            if d.startswith("fp2sim"):
                continue
            if os.path.isdir(os.path.join(path, d)):
                self._clean_descriptors_by_subfolder(path, d)
            else:
                if d.endswith(".h5"):
                    os.remove(os.path.join(path, d))

    def _clean_descriptors(self):
        self._clean_descriptors_by_subfolder(self.path, DESCRIPTORS_SUBFOLDER)

    def run(self):
        self.logger.debug("Cleaning descriptors by subfolder")
        self._clean_descriptors()


class Flusher(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        self.trained_dir = self.get_trained_dir()
        assert os.path.exists(self.output_dir)
        assert os.path.exists(self.trained_dir)

    def _remover(self, path):
        rm_dirs = []
        rm_files = []
        for root, dirs, files in os.walk(path):
            for filename in files:
                if filename.endswith(".json") or filename.endswith(".csv"):
                    continue
                else:
                    rm_files += [os.path.join(root, filename)]
            for dirname in dirs:
                if dirname.startswith("autogluon"):
                    rm_dirs += [os.path.join(root, dirname)]
                if dirname.startswith("kerastuner"):
                    rm_dirs += [os.path.join(root, dirname)]
        for f in rm_files:
            os.remove(f)
        for d in rm_dirs:
            shutil.rmtree(d)

    def _flush(self, path):
        self.logger.debug("Removing files descriptors folder in {0}".format(path))
        self._remover(os.path.join(path, DESCRIPTORS_SUBFOLDER))
        self.logger.debug("Removing files from estimators folder in {0}".format(path))
        self._remover(os.path.join(path, ESTIMATORS_SUBFOLDER))

    def run(self):
        self._flush(self.output_dir)
        self._flush(self.trained_dir)


class OutputToExcel(ZairaBase):
    def __init__(self, path, clean=False, flush=False):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_csv = os.path.join(self.path, OUTPUT_TABLE_FILENAME)
        self.performance_csv = os.path.join(self.path, PERFORMANCE_TABLE_FILENAME)
        self.output_xlsx = os.path.join(self.path, OUTPUT_XLSX_FILENAME)

    def run(self):
        df_o = pd.read_csv(self.output_csv)
        df_p = pd.read_csv(self.performance_csv)
        with pd.ExcelWriter(self.output_xlsx, mode="w", engine="openpyxl") as writer:
            df_o.to_excel(writer, sheet_name="Output", index=False)
            df_p.to_excel(writer, sheet_name="Performance", index=False)


class Finisher(ZairaBase):
    def __init__(self, path, clean=False, flush=False):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.clean = clean
        self.flush = flush

    def _clean_descriptors(self):
        Cleaner(path=self.path).run()

    def _flush(self):
        Flusher(path=self.path).run()

    def _predictions_file(self):
        shutil.copy(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME),
            os.path.join(self.path, OUTPUT_FILENAME),
        )

    def _output_table_file(self):
        shutil.copy(
            os.path.join(self.path, REPORT_SUBFOLDER, OUTPUT_TABLE_FILENAME),
            os.path.join(self.path, OUTPUT_TABLE_FILENAME),
        )

    def _performance_table_file(self):
        shutil.copy(
            os.path.join(self.path, REPORT_SUBFOLDER, PERFORMANCE_TABLE_FILENAME),
            os.path.join(self.path, PERFORMANCE_TABLE_FILENAME),
        )

    def _to_excel(self):
        OutputToExcel(path=self.path).run()

    def run(self):
        self.logger.debug("Finishing")
        self._predictions_file()
        self._output_table_file()
        self._performance_table_file()
        self._to_excel()
        if self.clean:
            self._clean_descriptors()
        if self.flush:
            self._flush()
