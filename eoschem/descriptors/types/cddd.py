from . import BaseDescriptorType
from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
from rdkit import Chem


class Cddd(BaseDescriptorType):
    def __init__(self):
        super().__init__()
        self.mdl = InferenceModel()

    def _calc(self, mols):
        smiles = preprocess_smiles([Chem.MolToSmiles(mol) for mol in mols])
        emb = self.mdl.seq_to_emb(smiles)
        return emb
