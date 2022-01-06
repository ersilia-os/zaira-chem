import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "../bidd-molmap/"))

import pandas as pd
import numpy as np

from molmap import MolMap
from molmap import feature
from molmap.model import load_model

file_name = sys.argv[1]
model_path = sys.argv[2]
path = os.path.dirname(file_name)

mp1 = MolMap(ftype="descriptor", metric="cosine").load(
    filename=os.path.join(root, "../data/descriptor.mp")
)

bitsinfo = feature.fingerprint.Extraction().bitsinfo
flist = bitsinfo[bitsinfo.Subtypes.isin(["PubChemFP"])].IDs.tolist()
mp2 = MolMap(ftype="fingerprint", fmap_type="scatter", flist=flist).load(
    filename=os.path.join(root, "../data/fingerprint.mp")
)

SMILES_COLUMN = "smiles"

data = pd.read_csv(file_name)

smiles_list = list(data[SMILES_COLUMN])

X1 = mp1.batch_transform(smiles_list)
X2 = mp2.batch_transform(smiles_list)

# Load regression
reg = load_model(os.path.join(model_path, "reg"))
reg_preds = reg.predict((X1, X2))[:, 0]
np.save(os.path.join(path, "reg_preds.npy"), reg_preds)

# Load classification
clf = load_model(os.path.join(model_path, "clf"))
clf_preds = clf.predict_proba((X1, X2))[:, 1]
np.save(os.path.join(path, "clf_preds.npy"), clf_preds)
