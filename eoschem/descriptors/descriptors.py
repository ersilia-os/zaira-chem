import numpy as np

from .types.maccs import Maccs
from .types.ecfp import Ecfp
from .types.signaturizer import Signaturizer


descriptor_factory = [
    Avalon(),
    Ecfp(),
    #Grover(),
    Maccs(),
    #Signaturizer()
]


class Adapter(object):

    def __init__(self, data):
        self._data = data
        self._functions = dict((d.name, d.calc) for d in descriptor_factory)

    @staticmethod
    def smiles_to_mols(query_smiles):
        mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs


class DescriptorsCalculator(object):

    def __init__(self, data):
        self.adapter = Adapter(data)
        self.data = data
        self._funcs = self._functions()

    def _functions(self):
        return self.adapter._functions

    def _calculate(self, n):
        func = self._funcs[n]
        res, idxs = func()
        X = np.zeros((len(self.data), np.array(res).shape[1])) # assume zeros for invalid smiles
        X[np.array(idxs)] = res
        return X

    def calculate_one(self, n):
        return self._calculate(n)

    def calculate(self):
        names = sorted([k for k, _ in self._funcs.items()])
        for n in names:
            X = self._calculate(n)
            yield X, n
