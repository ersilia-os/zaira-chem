import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA

from .. import ZairaBase
from ..descriptors.baseline import Fingerprinter
from ..tools.fpsim2.searcher import SimilaritySearcher, RandomSearcher
from ..tools.autogluon.multilabel import MultilabelPredictor, TabularDataset

from ..setup import COMPOUND_IDENTIFIER_COLUMN
from ..tools.molmap.molmap import SMILES_COLUMN
from ..vars import DATA_FILENAME, DATA_SUBFOLDER

MAX_N = 5000
MIN_N = 0
AUGMENTATION_FACTOR = 0.1

TANIMOTO_CUTOFF = 0.6
_TIME_BUDGET_SEC = 60


class Sampler(ZairaBase):

    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.reference_searcher = SimilaritySearcher()
        self.random_searcher = RandomSearcher()

    def has_available_nodes(self, G, query_smiles):
        for n in G.nodes():
            if n not in query_smiles:
                return True
        return False

    def _sample(self, smiles_list, cutoff, max_n=1000):
        query_smiles = set(smiles_list)
        G = nx.Graph()
        for smi in smiles_list:
            results = self.reference_searcher.search(smi, cutoff=cutoff)
            for r in results:
                G.add_edge(smi, r[1], weight=r[2])
        node_clashes = set()
        for e in G.edges(data=True):
            if e[2]["weight"] == 1:
                node_clashes.update([e[1]])
        for n in list(node_clashes):
            if n in query_smiles:
                continue
            else:
                G.remove_node(n)
        sampled = set()
        while self.has_available_nodes(G, query_smiles):
            for smi in smiles_list:
                edges = G.edges(smi, data=True)
                edges = dict((e[1],e[2]["weight"]) for e in edges if e[1] not in query_smiles)
                if not edges:
                    continue
                edge = sorted(edges.items(), key=lambda x: -x[1])[0]
                G.remove_node(edge[0])
                sampled.update([edge[0]])
                if len(sampled) >= max_n:
                    return sampled
            random.shuffle(smiles_list)
        return sampled

    def sample(self, smiles_list, n):
        ref = self._sample(smiles_list, TANIMOTO_CUTOFF, max_n=n)
        rnd = self.random_searcher.search(n-len(ref))
        sampled = list(ref.union(rnd))
        random.shuffle(sampled)
        return sampled


class Augmenter(ZairaBase):

    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.df_exp = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        self.n_orig = self.df_exp.shape[0]
        self.sampler = Sampler(self.path)
        self.smiles_exp = list(self.df_exp[SMILES_COLUMN])
        self.save_path = os.path.join(self.path, DATA_SUBFOLDER, "augmenter")
        self.time_limit = _TIME_BUDGET_SEC

    def _heuristic_n(self):
        ne = self.df_exp.shape[0]
        n = int(np.clip(ne*AUGMENTATION_FACTOR, MIN_N, MAX_N))
        self.logger.debug("Augmented samples: {0}".format(n))
        return n

    def _sample_smiles(self):
        n = self._heuristic_n()
        self.smiles_sampled = self.sampler.sample(smiles_list=self.smiles_exp, n=n)
        self.cids_sampled = ["VID{0}".format(i) for i in range(len(self.smiles_sampled))]
        self.all_smiles = self.smiles_exp + self.smiles_sampled

    def _get_df_xy(self):
        X = Fingerprinter().calculate(self.all_smiles)
        X = PCA(n_components=min(500, X.shape[0])).fit_transform(X)
        colnames = ["f-{0}".format(i) for i in range(X.shape[1])]
        df_xy = pd.DataFrame(X, columns = colnames)
        labels = []
        problem_types = []
        for col in list(self.df_exp.columns):
            if "reg_" in col and "_skip" not in col and "_aux" not in col:
                values = list(self.df_exp[col]) + [None]*(X.shape[0]-self.df_exp.shape[0])
                problem_types += ["regression"]
                labels += [col]
                df_xy[col] = values
                continue
            if "clf_" in col and "_skip" not in col and "_aux" not in col:
                values = list(self.df_exp[col]) + [None]*(X.shape[0]-self.df_exp.shape[0])
                problem_types += ["binary"]
                labels += [col]
                df_xy[col] = values
        self.df_xy = TabularDataset(df_xy)
        self.labels = labels
        self.problem_types = problem_types

    def _train_model(self):
        self.model = MultilabelPredictor(
            labels=self.labels,
            path=self.save_path,
            problem_types=self.problem_types,
            consider_labels_correlation=False
        )
        train_data = self.df_xy.iloc[:self.n_orig]
        unlabeled_data = self.df_xy.iloc[self.n_orig:]
        self.model.fit(
            train_data=train_data,
            #hyperparameters={"TRANSF": {}},
            hyperparameters="default",
            unlabeled_data=unlabeled_data[[c for c in list(unlabeled_data.columns) if c not in self.labels]],
            time_limit=self.time_limit,
            refit_full=True,
            presets="best_quality"
        )
        self.dfa = self.model.predict(unlabeled_data)
        self.dfa[SMILES_COLUMN] = self.smiles_sampled
        self.dfa[COMPOUND_IDENTIFIER_COLUMN] = self.cids_sampled
        
    def _assemble(self):
        dfe = self.df_exp
        dfa = self.dfa
        dfe["is_exp"] = 1
        dfa["is_exp"] = 0
        for c in list(dfe.columns):
            if "clf" in c and "_aux" not in c and "_skip" not in c:
                col = c
                if list(dfe[col]) == list(dfe["clf_aux"]):
                    break
        dfa["clf_aux"] = dfa[col]
        val_prop = np.sum(list(dfe["fld_val"]))/dfe.shape[0]
        val_a = [0]*dfa.shape[0]
        n = int(dfa.shape[0]*val_prop+1)
        idxs = random.sample([i for i in range(dfa.shape[0])], n)
        for i in idxs:
            val_a[i] = 1
        dfa["fld_val"] = val_a
        dfe = dfe.append(dfa, ignore_index=True)
        dfe.to_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME), index=False)

    def run(self):
        return
        self._sample_smiles()
        self._get_df_xy()
        self._train_model()
        self._assemble()