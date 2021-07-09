from . import BaseDescriptorType
from rdkit import Chem
from signaturizer import Signaturizer as _Signaturizer

dataset = "GLOBAL"


class Signaturizer(BaseDescriptorType):
    def __init__(self):
        super().__init__()
        self.dataset = dataset


    def _calc(self, mols):
        sign = _Signaturizer(dataset)
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        return sign.predict(smiles).signature
