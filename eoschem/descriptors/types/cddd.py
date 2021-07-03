from cddd.inference import InferenceModel
from cddd.preprocessing import preprocess_smiles
from rdkit import Chem


class Cddd(object):
    def __init__(self):
        self.name = "cddd"
        self.mdl = InferenceModel()

    def calc(self, mols):
        smiles = preprocess_smiles([Chem.MolToSmiles(mol) for mol in mols])
        emb = self.mdl.seq_to_emb(smiles)
        return emb
