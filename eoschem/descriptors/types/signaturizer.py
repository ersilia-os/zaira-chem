from rdkit import Chem
from signaturizer import Signaturizer as _Signaturizer

dataset = "GLOBAL"


class Signaturizer(mols):

    def __init__(self):
        self.name = "signaturizer"
        self.dataset = dataset

    def calc(self, mols):
        sign = _Signaturizer(dataset)
        smiles = [Chem.MolToSmiles(mol) for mol in mols]
        return sign.predict(smiles).signature
