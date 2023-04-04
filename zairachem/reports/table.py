import os
import pandas as pd
import collections
from rdkit import Chem

from . import BaseTable
from .fetcher import ResultsFetcher

from ..vars import OUTPUT_TABLE_FILENAME
from ..vars import PERFORMANCE_TABLE_FILENAME
from ..vars import REPORT_SUBFOLDER


class PerformanceTable(BaseTable, ResultsFetcher):
    def __init__(self, path):
        BaseTable.__init__(self, path=path)
        ResultsFetcher.__init__(self, path=path)
        self.is_clf = self.has_clf_data()

    def _individual_performances(self):
        if self.is_clf:
            tasks = self.get_clf_tasks()
        else:
            tasks = self.get_reg_tasks()
        task = tasks[0]
        df_te = self._read_individual_estimator_results(task)
        df_tr = self._read_individual_estimator_results_train(task)
        columns = list(df_te.columns)
        for col in columns:
            y_pred_test = list(df_te[col])
            y_pred_train = list(df_tr[col])
            if self.is_clf:
                y_true_train = list(self.get_actives_inactives_trained())
                y_true_test = list(self.get_actives_inactives())
                data = self.classification_performance_report(
                    y_true_train, y_pred_train, y_true_test, y_pred_test
                )
            else:
                # TODO
                data = self.regression_performance_report(
                    y_true_train, y_pred_train, y_true_test, y_pred_test
                )
            yield (col, data)

    def _general_performance(self):
        if self.is_clf:
            y_true_train = list(self.get_actives_inactives_trained())
            y_true_test = list(self.get_actives_inactives())
            y_pred_train = list(self.get_pred_proba_clf_trained())
            y_pred_test = list(self.get_pred_proba_clf())
            data = self.classification_performance_report(
                y_true_train, y_pred_train, y_true_test, y_pred_test
            )
        else:
            data = None  # TODO
        return data

    def run(self):
        data = collections.defaultdict(list)
        d = self._general_performance()
        data["model"] += ["pooled"]
        if d is None:
            return
        for k, v in d.items():
            data[k] += [v]
        for col, d_ in self._individual_performances():
            data["model"] += [col]
            for k, v in d_.items():
                data[k] += [v]
        data = pd.DataFrame(data)
        data.to_csv(
            os.path.join(self.path, REPORT_SUBFOLDER, PERFORMANCE_TABLE_FILENAME),
            index=False,
        )


class OutputTable(BaseTable, ResultsFetcher):
    def __init__(self, path):
        BaseTable.__init__(self, path=path)
        ResultsFetcher.__init__(self, path=path)
        self.is_clf = self.has_clf_data()

    def __smiles_to_inchikey(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise Exception(
                "The SMILES string: %s is not valid or could not be converted to an InChIKey"
                % smiles
            )
        inchi = Chem.rdinchi.MolToInchi(mol)[0]
        if inchi is None:
            raise Exception("Could not obtain InChI")
        inchikey = Chem.rdinchi.InchiToInchiKey(inchi)
        return inchikey

    def _get_identifier_column(self):
        return self.get_original_identifiers()

    def _get_input_smiles_column(self):
        return self.get_original_smiles()

    def _get_inchikey_column(self):
        inchikeys = [self.__smiles_to_inchikey(smiles) for smiles in self.get_smiles()]
        return self.map_to_original(inchikeys)

    def _get_smiles_column(self):
        smiles = self.get_smiles()
        return self.map_to_original(smiles)

    def _get_true_value_column(self):
        if self.is_clf:
            values = self.get_true_clf()
            return self.map_to_original(values)
        else:
            return None

    def _get_pred_value_column(self):
        try:
            values = self.get_pred_proba_clf()
            return self.map_to_original(values)
            # values = self.get_pred_reg_trans()
        except:
            return None

    def _get_ensemble_predictions_columns(self):
        try:
            tasks = self.get_clf_tasks()
            print(tasks)
        except:
            tasks = self.get_reg_tasks()
        task = tasks[0]
        df = self._read_individual_estimator_results(task)
        columns = list(df.columns)
        for col in columns:
            v = list(df[col])
            v = self.map_to_original(v)
            yield (col, v)

    def _get_manifolds_columns(self):
        umap = self.get_projections_umap()
        pca = self.get_projections_pca()
        data = {"umap-0": umap[0], "umap-1": umap[1], "pca-0": pca[0], "pca-1": pca[1]}
        df = pd.DataFrame(data)
        columns = list(df.columns)
        for col in columns:
            v = list(df[col])
            v = self.map_to_original(v)
            yield (col, v)

    def _get_basic_properties_columns(self):
        df = self.get_basic_properties()
        columns = list(df.columns)
        for col in columns:
            v = list(df[col])
            v = self.map_to_original(v)
            yield (col, v)

    def _get_similarity_to_training_set_columns(self):
        df = self.get_tanimoto_similarities_to_training_set()
        columns = list(df.columns)
        for col in columns:
            v = list(df[col])
            v = self.map_to_original(v)
            yield (col, v)

    def run(self):
        data = {}
        data["identifier"] = self._get_identifier_column()
        data["input-smiles"] = self._get_input_smiles_column()
        data["inchikey"] = self._get_inchikey_column()
        data["smiles"] = self._get_smiles_column()
        data["true-value"] = self._get_true_value_column()
        data["pred-value"] = self._get_pred_value_column()
        for k, v in self._get_ensemble_predictions_columns():
            data[k] = v
        for k, v in self._get_manifolds_columns():
            data[k] = v
        for k, v in self._get_basic_properties_columns():
            data[k] = v
        for k, v in self._get_similarity_to_training_set_columns():
            data[k] = v
        data = pd.DataFrame(data)
        data.to_csv(
            os.path.join(self.path, REPORT_SUBFOLDER, OUTPUT_TABLE_FILENAME),
            index=False,
        )
