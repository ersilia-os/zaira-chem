from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcTPSA

from ..automl.classic import classic_featurizer


class BasicProperties(object):
    def __init__(self):
        pass

    def calculate(self, smiles_list):
        df = classic_featurizer(smiles_list)
        df = df.rename(
            columns={
                "number of hydrogen bond donor": "nHBD",
                "number of hydrogen bond acceptor": "nHBA",
                "Wildman-Crippen LogP": "cLogP",
                "number of heteroatoms": "nHeteroAtoms",
                "ring count": "RingCount",
                "number of rotatable bonds": "nRotatableBonds",
                "aromatic bonds count": "nAromaticBonds",
                "acidic group count": "nAcidicGroup",
                "basic group count": "nBasicGroup",
                "atomic polarizability": "AtomicPolarizability",
            }
        )
        mols_list = [Chem.MolFromSmiles(m) for m in smiles_list]
        df["MolWt"] = [MolWt(x) for x in mols_list]
        df["TPSA"] = [CalcTPSA(x) for x in mols_list]
        return df
