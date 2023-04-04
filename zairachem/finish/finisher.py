import os
import shutil
import pandas as pd
import glob

from ..vars import (
    DESCRIPTORS_SUBFOLDER,
    POOL_SUBFOLDER,
    ESTIMATORS_SUBFOLDER,
    REPORT_SUBFOLDER,
    OUTPUT_FILENAME,
    OUTPUT_TABLE_FILENAME,
    PERFORMANCE_TABLE_FILENAME,
    OUTPUT_XLSX_FILENAME,
    APPLICABILITY_SUBFOLDER,
    DATA_SUBFOLDER,
    DATA_FILENAME,
)
from .. import ZairaBase
from ..setup import RAW_INPUT_FILENAME
from ..estimators import RESULTS_MAPPED_FILENAME, RESULTS_UNMAPPED_FILENAME


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


class Anonymizer(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self._empty_string = "NA"

    def _remove_file_if_exists(self, file_name):
        if os.path.exists(file_name):
            os.remove(file_name)

    def _replace_sensitive_columns(self, file_name):
        columns = ["input-smiles", "smiles", "inchikey"]
        df = pd.read_csv(file_name)
        df_columns = set(list(df.columns))
        for col in columns:
            if col in df_columns:
                df[col] = self._empty_string
        df.to_csv(file_name, index=False)

    def _remove_sensitive_columns(self, file_name):
        desc_columns = [
            "nHBD",
            "nHBA",
            "cLogP",
            "nHeteroAtoms",
            "RingCount",
            "nRotatableBonds",
            "nAromaticBonds",
            "nAcidicGroup",
            "nBasicGroup",
            "AtomicPolarizability",
            "MolWt",
            "TPSA",
        ]
        prefixes = ["umap-", "pca-", "lolp-", "exact_", "sim_", "train_smiles_"]
        df = pd.read_csv(file_name)
        columns = list(df.columns)
        to_remove = []
        for col in columns:
            if col in desc_columns:
                to_remove += [col]
                continue
            for pref in prefixes:
                if col.startswith(pref):
                    to_remove += [col]
        to_remove = list(set(to_remove))
        df.drop(columns=to_remove, inplace=True)
        df.to_csv(file_name, index=False)

    def _replace_first_last_descriptors(self,file_name):
        df = pd.read_csv(file_name)
        df.loc[0:1, df.columns[1:]] = self._empty_string
        df.to_csv(file_name, index=False)


    def _remove_raw_input(self):
        file_name = os.path.join(self.path, RAW_INPUT_FILENAME + ".csv")
        self._remove_file_if_exists(file_name)

    def _remove_output_table_xlsx(self):
        file_name = os.path.join(self.path, OUTPUT_XLSX_FILENAME)
        self._remove_file_if_exists(file_name)

    def _remove_applicability_files(self):
        for fn in os.listdir(os.path.join(self.path, APPLICABILITY_SUBFOLDER)):
            fn = os.path.join(self.path, APPLICABILITY_SUBFOLDER, fn)
            self._remove_file_if_exists(fn)

    def _clear_all_sensitive_columns(self):
        self._replace_sensitive_columns(
            os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        )
        self._replace_sensitive_columns(os.path.join(self.path, OUTPUT_FILENAME))
        self._replace_sensitive_columns(os.path.join(self.path, OUTPUT_TABLE_FILENAME))
        self._remove_sensitive_columns(os.path.join(self.path, OUTPUT_TABLE_FILENAME))

    def _clear_descriptors(self):
        Cleaner(path=self.path).run()
        subfolder = os.path.join(self.path, DESCRIPTORS_SUBFOLDER)
        for d in os.listdir(subfolder):
            if d.startswith("fp2sim"):
                self._remove_file_if_exists(os.path.join(subfolder, d))
        for file_path in glob.iglob(subfolder+"/**", recursive=True):
            file_name = file_path.split("/")[-1]
            if file_name == "raw_summary.csv":
                self._replace_first_last_descriptors(file_path)
                

    def _clear_estimators(self):
        subfolder = os.path.join(self.path, ESTIMATORS_SUBFOLDER).rstrip("/")
        for file_path in glob.iglob(subfolder + "/**", recursive=True):
            file_name = file_path.split("/")[-1]
            if (
                file_name == RESULTS_MAPPED_FILENAME
                or file_name == RESULTS_UNMAPPED_FILENAME
            ):
                self._replace_sensitive_columns(file_path)
        self._remove_file_if_exists(os.path.join(subfolder, "manifolds", "data.csv"))
        self._remove_file_if_exists(
            os.path.join(subfolder, "reference_embedding", "data.csv")
        )

    def _clear_pool(self):
        self._remove_sensitive_columns(
            os.path.join(self.path, POOL_SUBFOLDER, "data.csv")
        )
        self._replace_sensitive_columns(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_MAPPED_FILENAME)
        )
        self._replace_sensitive_columns(
            os.path.join(self.path, POOL_SUBFOLDER, RESULTS_UNMAPPED_FILENAME)
        )

    def _clear_report(self):
        self._remove_file_if_exists(
            os.path.join(
                self.path, REPORT_SUBFOLDER, "tanimoto-similarity-to-train.png"
            )
        )
        self._remove_file_if_exists(
            os.path.join(self.path, REPORT_SUBFOLDER, "projection-pca.png")
        )
        self._remove_file_if_exists(
            os.path.join(self.path, REPORT_SUBFOLDER, "projection-umap.png")
        )
        self._remove_file_if_exists(
            os.path.join(self.path, REPORT_SUBFOLDER, OUTPUT_TABLE_FILENAME)
        )

    def run(self):
        self._remove_raw_input()
        self._remove_output_table_xlsx()
        self._remove_applicability_files()
        self._clear_all_sensitive_columns()
        self._clear_descriptors()
        self._clear_estimators()
        self._clear_pool()
        self._clear_report()


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
        if not os.path.exists(self.performance_csv):
            df_p = None
        else:
            df_p = pd.read_csv(self.performance_csv)
        with pd.ExcelWriter(self.output_xlsx, mode="w", engine="openpyxl") as writer:
            df_o.to_excel(writer, sheet_name="Output", index=False)
            if df_p is not None:
                df_p.to_excel(writer, sheet_name="Performance", index=False)


class Finisher(ZairaBase):
    def __init__(self, path, clean=False, flush=False, anonymize=False):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.clean = clean
        self.flush = flush
        self.anonymize = anonymize

    def _clean_descriptors(self):
        Cleaner(path=self.path).run()

    def _flush(self):
        Flusher(path=self.path).run()
    
    def _anonymize(self):
        Anonymizer(path=self.path).run()

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
        filename = os.path.join(self.path, REPORT_SUBFOLDER, PERFORMANCE_TABLE_FILENAME)
        if not os.path.exists(filename):
            return
        shutil.copy(
            filename,
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
        if self.anonymize:
            self._anonymize()
