import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "../bidd-molmap/"))

import pandas as pd
import numpy as np

from molmap.model import load_model
from utils import descriptors_molmap, fingerprints_molmap

file_name = sys.argv[1]
model_path = sys.argv[2]
path = os.path.dirname(file_name)

mp1 = descriptors_molmap()
mp2 = fingerprints_molmap()

SMILES_COLUMN = "smiles"

data = pd.read_csv(file_name)

smiles_list = list(data[SMILES_COLUMN])

X1 = mp1.batch_transform(smiles_list)
X2 = mp2.batch_transform(smiles_list)

# Load regression
if os.path.exists(os.path.join(model_path, "reg")):
    reg = load_model(os.path.join(model_path, "reg"))
    reg_preds = reg.predict((X1, X2))[:, 0]
    np.save(os.path.join(path, "reg_preds.npy"), reg_preds)

# Load classification
if os.path.exists(os.path.join(model_path, "clf")):
    clf = load_model(os.path.join(model_path, "clf"))
    clf_preds = clf.predict_proba((X1, X2))[:, 1]
    np.save(os.path.join(path, "clf_preds.npy"), clf_preds)
