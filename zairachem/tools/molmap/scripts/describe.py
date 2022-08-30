import pandas as pd
import sys
from ersilia import ErsiliaModel
import numpy as np


file_name = sys.argv[1]
x1_path = sys.argv[2]
x2_path = sys.argv[3]

SMILES_COLUMN = "smiles"
GPUID = 1

data = pd.read_csv(file_name)

smiles_list = list(data[SMILES_COLUMN])

with ErsiliaModel("bidd-molmap") as mdl:
    X1 = mdl.descriptors(input=smiles_list, output="numpy")
    X1 = X1.reshape(X1.shape[0], 37, 37, 1)
    X2 = mdl.fingerprints(input=smiles_list, output="numpy")
    X2 = X2.reshape(X2.shape[0], 37, 36, 1)

with open(x1_path, "wb") as f:
    np.save(f, X1)

with open(x2_path, "wb") as f:
    np.save(f, X2)
