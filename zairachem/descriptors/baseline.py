import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rd
from ersilia import ErsiliaModel
from tqdm import tqdm
import h5py
import os

from zairachem.descriptors.treated import FullLineSimilarityImputer
from zairachem.vars import DESCRIPTORS_SUBFOLDER

from .. import ZairaBase

RADIUS = 3
NBITS = 2048
DTYPE = np.int8


def clip_sparse(vect, nbits):
    l = [0] * nbits
    for i, v in vect.GetNonzeroElements().items():
        l[i] = v if v < 255 else 255
    return l


class _Fingerprinter(object):
    def __init__(self):
        self.nbits = NBITS
        self.radius = RADIUS

    def calc(self, mol):
        v = rd.GetHashedMorganFingerprint(mol, radius=self.radius, nBits=self.nbits)
        return clip_sparse(v, self.nbits)


class Fingerprinter(object):
    def __init__(self):
        self.fingerprinter = _Fingerprinter()

    def _calculate(self, smiles_list):
        X = np.zeros((len(smiles_list), NBITS), np.uint8)
        for i, smi in tqdm(enumerate(smiles_list)):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            X[i, :] = self.fingerprinter.calc(mol)
        return X

    def calculate(self, smiles_list, output_h5=None):
        X = self._calculate(smiles_list)
        if output_h5 is None:
            return X
        keys = ["mol-{0}".format(i) for i in range(X.shape[0])]
        features = ["f-{0}".format(i) for i in range(X.shape[1])]
        inputs = smiles_list
        with h5py.File(output_h5, "w") as f:
            f.create_dataset("Keys", data=keys, dtype=h5py.string_dtype())
            f.create_dataset("Features", data=features, dtype=h5py.string_dtype())
            f.create_dataset("Inputs", data=inputs, dtype=h5py.string_dtype())
            f.create_dataset("Values", data=X)


class Embedder(ZairaBase):
    def __init__(self):
        ZairaBase.__init__(self)
        self.dim = 5000
        self.model = "grover-embedding"

    def _calculate(self, smiles_list, output_h5):
        if output_h5 is None:
            with ErsiliaModel(self.model) as mdl:
                X = mdl.api(api_name=None, input=smiles_list, output="numpy")
            return X
        else:
            with ErsiliaModel(self.model) as mdl:
                mdl.api(api_name=None, input=smiles_list, output=output_h5)

    def calculate(self, smiles_list, output_h5=None):
        X = self._calculate(smiles_list, output_h5)
        if X is None:
            with h5py.File(output_h5, "r") as f:
                X = f["Values"][:]
        imp = FullLineSimilarityImputer()
        trained_path = self.get_trained_dir()
        path = self.get_output_dir()
        if not self.is_predict():
            imp.fit(X, smiles_list)
            X = imp.transform(X, smiles_list)
            imp.save(
                os.path.join(
                    path, DESCRIPTORS_SUBFOLDER, "{0}.joblib".format(imp._prefix)
                )
            )
        else:
            imp = imp.load(
                os.path.join(
                    trained_path,
                    DESCRIPTORS_SUBFOLDER,
                    "{0}.joblib".format(imp._prefix),
                )
            )
            X = imp.transform(X, smiles_list)
        if output_h5 is None:
            return X
        else:
            with h5py.File(output_h5, "r") as f:
                keys = f["Keys"][:]
                inputs = f["Inputs"][:]
                features = f["Features"][:]
            os.remove(output_h5)
            with h5py.File(output_h5, "w") as f:
                f.create_dataset("Keys", data=keys)
                f.create_dataset("Features", data=features)
                f.create_dataset("Inputs", data=inputs)
                f.create_dataset("Values", data=X)
