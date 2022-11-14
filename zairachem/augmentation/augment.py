import os
import shutil
import random
import numpy as np
import pandas as pd
from lazyqsar.binary.morgan import MorganBinaryClassifier
from .. import ZairaBase
from ..setup import COMPOUND_IDENTIFIER_COLUMN
from ..tools.molmap.molmap import SMILES_COLUMN
from ..vars import (
    DATA_FILENAME,
    DATA_SUBFOLDER,
    REFERENCE_FILENAME,
    DATA_AUGMENTED_FILENAME,
)

_MIN_CLASS_N = 100
_MIN_CLASS_IMBALANCE = 0.25
_SELF_TRAINED_MAX_ITER = 10

_LAZY_QSAR_TIME_BUDGET_SEC = 20


class SelfTrainedClassifierLabeler(object):
    def __init__(self, max_iter=10):
        self.max_iter = max_iter

    def run(self, smiles, y, unlabeled_smiles, num_samples_0, num_samples_1):
        assert num_samples_0 + num_samples_1 < len(unlabeled_smiles)
        n_orig_1 = int(np.sum(y))
        n_orig_0 = len(y) - n_orig_1
        y = list(y)
        num_samples_0_per_iter = int(num_samples_0 / (self.max_iter - 1))
        num_samples_1_per_iter = int(num_samples_1 / (self.max_iter - 1))
        sampled_0 = 0
        sampled_1 = 0
        iter_round = [-1] * len(smiles)
        for iter_i in range(self.max_iter):
            print(
                "Iteration:",
                iter_i,
                "Samples:",
                len(smiles),
                "Positives:",
                np.sum(y),
                "Negatives:",
                len(y) - np.sum(y),
                "Unlabeled Samples:",
                len(unlabeled_smiles),
            )
            mdl = MorganBinaryClassifier(
                automl=False, time_budget_sec=_LAZY_QSAR_TIME_BUDGET_SEC
            )
            mdl.fit(smiles, y)
            y_hat = mdl.predict_proba(unlabeled_smiles)
            sort_idxs_0 = np.argsort(y_hat[:, 0])[::-1]
            sort_idxs_1 = np.argsort(y_hat[:, 1])[::-1]
            append_smiles = []
            append_y = []
            new_unlabeled_smiles = []
            if sampled_0 < num_samples_0 and num_samples_0_per_iter > 0:
                yes_idxs = set(sort_idxs_0[:num_samples_0_per_iter])
                for i in sort_idxs_0:
                    if i in yes_idxs:
                        append_smiles += [unlabeled_smiles[i]]
                        append_y += [0]
                        sampled_0 += 1
                    else:
                        new_unlabeled_smiles += [unlabeled_smiles[i]]
            if sampled_1 < num_samples_1 and num_samples_1_per_iter > 0:
                yes_idxs = set(sort_idxs_1[:num_samples_1_per_iter])
                for i in sort_idxs_1:
                    if i in yes_idxs:
                        append_smiles += [unlabeled_smiles[i]]
                        append_y += [1]
                        sampled_1 += 1
                    else:
                        new_unlabeled_smiles += [unlabeled_smiles[i]]
            new_unlabeled_smiles = list(
                set(new_unlabeled_smiles).difference(append_smiles)
            )
            smiles += append_smiles
            y += append_y
            iter_round += [iter_i] * len(append_smiles)
            unlabeled_smiles = new_unlabeled_smiles
        count_0 = 0
        count_1 = 0
        expected_0 = num_samples_0 + n_orig_0
        expected_1 = num_samples_1 + n_orig_1
        smiles_ = []
        y_ = []
        iter_round_ = []
        for i, v in enumerate(y):
            if v == 0:
                if count_0 >= expected_0:
                    continue
                smiles_ += [smiles[i]]
                y_ += [y[i]]
                iter_round_ += [iter_round[i]]
                count_0 += 1
            else:
                if count_1 >= expected_1:
                    continue
                smiles_ += [smiles[i]]
                y_ += [y[i]]
                iter_round_ += [iter_round[i]]
                count_1 += 1
        smiles = smiles_
        y = y_
        iter_round = iter_round_
        print(
            "Iteration:",
            iter_i,
            "Samples:",
            len(smiles),
            "Positives:",
            np.sum(y),
            "Negatives:",
            len(y) - np.sum(y),
            "Unlabeled Samples:",
            len(unlabeled_smiles),
        )
        return smiles, y, iter_round


class ClassifierAugmenter(ZairaBase):
    def __init__(self, path):
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.df_exp = pd.read_csv(
            os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        )
        self.df_ref = pd.read_csv(
            os.path.join(self.path, DATA_SUBFOLDER, REFERENCE_FILENAME)
        )
        print("Experimental data:", self.df_exp.shape[0])
        print("Reference data", self.df_ref.shape[0])
        self.save_path = os.path.join(self.path, DATA_SUBFOLDER, "augmenter")
        self.smiles_exp = list(self.df_exp[SMILES_COLUMN])
        self.smiles_unlabeled = list(self.df_ref[SMILES_COLUMN])
        self._get_relevant_column_name()
        self.y_exp = np.array(list(self.df_exp[self._relevant_column]))
        self.n_exp_1 = np.sum(self.y_exp == 1)
        self.n_exp_0 = np.sum(self.y_exp == 0)
        self.n_exp = len(self.y_exp)

    def _get_relevant_column_name(self):
        for c in list(self.df_exp.columns):
            if "clf" in c and "_aux" not in c and "_skip" not in c:
                col = c
                if list(self.df_exp[col]) == list(self.df_exp["clf_aux"]):
                    break
        self._relevant_column = col

    def _needs_sampling(self):
        if self.n_exp_1 < _MIN_CLASS_N:
            return True
        if self.n_exp_0 < _MIN_CLASS_N:
            return True
        if self.n_exp_1 > self.n_exp_0:
            return True
        if (self.n_exp_1 / self.n_exp) < _MIN_CLASS_IMBALANCE:
            return True
        return False

    def _ask_num_samples(self):
        print("Original 0:", self.n_exp_0, "Original 1:", self.n_exp_1)
        if self.n_exp_1 < _MIN_CLASS_N:
            n_1 = _MIN_CLASS_N
        else:
            n_1 = self.n_exp_1
        if self.n_exp_0 < _MIN_CLASS_N:
            n_0 = _MIN_CLASS_N
        else:
            n_0 = self.n_exp_0
        if n_1 > n_0:
            n_0 = n_1
        if (n_1 / (n_0 + n_1)) < _MIN_CLASS_IMBALANCE:
            n_1 = int((_MIN_CLASS_IMBALANCE * n_0) / (1 - _MIN_CLASS_IMBALANCE))
        ask_0 = n_0 - self.n_exp_0
        ask_1 = n_1 - self.n_exp_1
        print("Original 0:", self.n_exp_0, "Original 1:", self.n_exp_1)
        print("Ask 0:", ask_0, "Ask 1:", ask_1)
        return ask_0, ask_1

    def _self_trained_classifier(self, num_0, num_1):
        mdl = SelfTrainedClassifierLabeler(max_iter=_SELF_TRAINED_MAX_ITER)
        smiles, y, iter_round = mdl.run(
            smiles=self.smiles_exp,
            y=self.y_exp,
            unlabeled_smiles=self.smiles_unlabeled,
            num_samples_0=num_0,
            num_samples_1=num_1,
        )
        sampled_smiles = []
        sampled_y = []
        sampled_cids = []
        j = 0
        for i, ir in enumerate(iter_round):
            if ir >= 0:
                sampled_smiles += [smiles[i]]
                sampled_y += [y[i]]
                sampled_cids += ["VID{0}".format(j)]
                j += 1
        self.df_smp = pd.DataFrame(
            {
                COMPOUND_IDENTIFIER_COLUMN: sampled_cids,
                SMILES_COLUMN: sampled_smiles,
                self._relevant_column: sampled_y,
                "clf_aux": sampled_y,
            }
        )

    def _assemble(self):
        dfe = self.df_exp
        dfs = self.df_smp
        dfe["is_exp"] = 1
        dfs["is_exp"] = 0
        val_prop = np.sum(list(dfe["fld_val"])) / dfe.shape[0]
        val_a = [0] * dfs.shape[0]
        n = int(dfs.shape[0] * val_prop + 1)
        idxs = random.sample([i for i in range(dfs.shape[0])], n)
        for i in idxs:
            val_a[i] = 1
        dfs["fld_val"] = val_a
        dfe = dfe.append(dfs, ignore_index=True)
        dfe.to_csv(
            os.path.join(self.path, DATA_SUBFOLDER, DATA_AUGMENTED_FILENAME),
            index=False,
        )

    def run(self):
        if self._needs_sampling():
            num_0, num_1 = self._ask_num_samples()
            self._self_trained_classifier(num_0=num_0, num_1=num_1)
            self._assemble()
        else:
            shutil.copy(
                os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME),
                os.path.join(self.path, DATA_SUBFOLDER, DATA_AUGMENTED_FILENAME),
            )


class Augmenter(ZairaBase):
    def __init__(self, path):
        self._orig_path = path
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.df_exp = pd.read_csv(
            os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME)
        )

    def is_clf(self):
        for col in list(self.df_exp.columns):
            if col.startswith("clf_") and not col.endswith("_skip"):
                return True
        return False

    def run(self):
        if self.is_clf():
            self.logger.debug("Classification task")
            aug = ClassifierAugmenter(path=self._orig_path)
            aug.run()
            return
        else:
            self.logger.debug("Not a classification")
            return
