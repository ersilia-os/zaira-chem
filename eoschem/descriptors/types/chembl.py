import os
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import onnxruntime as rt

FP_SIZE = 1024
RADIUS = 2
EXAMPLE = 'CN(C)CCc1c[nH]c2ccc(C[C@H]3COC(=O)N3)cc12'


class Chembl(object):

    def __init__(self):
        self.model_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "chembl_28_multitask.onnx"))
        self.ort_session = rt.InferenceSession(self.model_path)
        self._work_out_targets()

    def _work_out_targets(self):
        descs = self._calc_morgan_fp(Chem.MolFromSmiles(EXAMPLE))
        ort_inputs = {self.ort_session.get_inputs()[0].name: descs}
        preds = self.ort_session.run(None, ort_inputs)
        preds = self._format_preds(preds, [o.name for o in self.ort_session.get_outputs()])
        targets = []
        for p in preds:
            targets += [p[0]]
        self.targets = sorted(targets)
        self.target_idxs = dict((k, i) for i,k in enumerate(self.targets))

    def _calc_morgan_fp(self, mol):
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, RADIUS, nBits=FP_SIZE)
        a = np.zeros((0,), dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, a)
        return a

    def _format_preds(self, preds, targets):
        preds = np.concatenate(preds).ravel()
        np_preds = [(tar, pre) for tar, pre in zip(targets, preds)]
        dt = [('chembl_id','|U20'), ('pred', '<f4')]
        np_preds = np.array(np_preds, dtype=dt)
        np_preds[::-1].sort(order='pred')
        return np_preds

    def calc(self, mols):
        fps = []
        for mol in mols:
            descs = self._calc_morgan_fp(mol)
            ort_inputs = {self.ort_session.get_inputs()[0].name: descs}
            preds = self.ort_session.run(None, ort_inputs)
            preds = self._format_preds(preds, [o.name for o in self.ort_session.get_outputs()])
            fp = np.zeros(len(self.targets))
            for p in preds:
                fp[self.target_idxs[p[0]]] = p[1]
            fps += [fp]
        X = np.array(fps)
        return X
