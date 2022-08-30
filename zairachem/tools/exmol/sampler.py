from exmol import run_stoned
from tqdm import tqdm
import numpy as np
import time
import random
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from syba.syba import SybaClassifier


class StonedSampler(object):
    def __init__(self, max_mutations=2, min_mutations=1):
        self.max_mutations = max_mutations
        self.min_mutations = min_mutations

    def sample(self, smiles, n):
        return run_stoned(
            smiles,
            num_samples=n,
            max_mutations=self.max_mutations,
            min_mutations=self.min_mutations,
        )


class StonedBatchSampler(object):
    def __init__(
        self,
        min_similarity=0.6,
        max_similarity=0.9,
        scorer=None,
        inflation=2,
        time_budget_sec=60,
    ):
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.sampler = StonedSampler(max_mutations=5, min_mutations=1)
        if scorer is None:
            self.scorer = SybaClassifier()
            self.scorer.fitDefaultScore()
        else:
            self.scorer = scorer
        self.inflation = inflation
        self.time_budget_sec = time_budget_sec
        self.elapsed_time = 0
        self.finished = False

    def _sample(self, smiles_list, n):
        random.shuffle(smiles_list)
        n_individual = int(np.clip(self.inflation * n / len(smiles_list), 100, 1000))
        available_time = int((self.time_budget_sec - self.elapsed_time)) + 1
        samples_per_sec = 100
        estimated_time = len(smiles_list) / samples_per_sec
        if estimated_time > available_time:
            n_individual = 10
        sampled_smiles = []
        sampled_sim = []
        for smi in tqdm(smiles_list):
            t0 = time.time()
            sampled = self.sampler.sample(smi, n_individual)
            sampled_smiles += sampled[0]
            sampled_sim += sampled[1]
            t1 = time.time()
            dt = t1 - t0
            self.elapsed_time += dt
            if self.elapsed_time > self.time_budget_sec:
                self.finished = True
                break
        smiles = []
        for smi, sim in zip(sampled_smiles, sampled_sim):
            if sim < self.min_similarity or sim > self.max_similarity:
                continue
            smiles += [smi]
        n = int(len(smiles) / self.inflation + 1)
        smiles = self._select_by_similarity(smiles)
        smiles = self._select_by_score(smiles, n)
        return set(smiles)

    def _select_by_score(self, smiles, n):
        smiles = list(smiles)
        scores = [self.scorer.predict(smi) for smi in tqdm(smiles)]
        df = pd.DataFrame({"smiles": smiles, "score": scores})
        return list(df.sort_values(by="score").tail(n)["smiles"])

    def _select_by_similarity(self, smiles):
        sel_smiles = []
        for smi in tqdm(smiles):
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.seed_fps)
            sim = np.max(sims)
            if sim < self.min_similarity or sim > self.max_similarity:
                continue
            sel_smiles += [smi]
        return sel_smiles

    def sample(self, smiles_list, n):
        self.seed_smiles = list(smiles_list)
        self.seed_fps = [
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2)
            for smi in self.seed_smiles
        ]
        smiles = set(smiles_list)
        sampled_smiles = set()
        for i in range(n):
            new_smiles = self._sample(list(smiles), n)
            sampled_smiles.update(new_smiles)
            smiles.update(new_smiles)
            if self.finished:
                break
        smiles = list(sampled_smiles)
        smiles = self._select_by_similarity(smiles)
        if len(smiles) > n:
            smiles = self._select_by_score(smiles, n)
        self.elapsed_time = 0
        self.finished = False
        return smiles
