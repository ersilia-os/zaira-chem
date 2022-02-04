import sys
import os

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "../bidd-molmap/"))

import pandas as pd
import numpy as np

import molmap
from molmap.model import save_model
from molmap.model import RegressionEstimator, MultiClassEstimator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

file_name = sys.argv[1]
model_path = sys.argv[2]

metric = 'cosine'
method = 'umap'
n_neighbors = 30
min_dist = 0.1

mp_name = os.path.join(root, '../data/descriptor.mp')
mp1 = molmap.MolMap(ftype = 'descriptor', metric = metric, flist = [])
mp1.fit(method = method, n_neighbors = n_neighbors, min_dist = min_dist)

mp_name = os.path.join(root, '../data/fingerprint.mp')
bitsinfo = molmap.feature.fingerprint.Extraction().bitsinfo
flist = bitsinfo[bitsinfo.Subtypes.isin(['MACCSFP', 'PharmacoErGFP','PubChemFP'])].IDs.tolist()
mp2 = molmap.MolMap(ftype = 'fingerprint', metric = metric, flist = flist)
mp2.fit(method = method, n_neighbors = n_neighbors, min_dist = min_dist)


SMILES_COLUMN = "smiles"
GPUID = 1

data = pd.read_csv(file_name)

smiles_list = list(data[SMILES_COLUMN])

X1 = mp1.batch_transform(smiles_list)
X2 = mp2.batch_transform(smiles_list)

reg_columns = [
    c
    for c in list(data.columns)
    if "reg_" in c and "_skip" not in c and "_aux" not in c
]
clf_columns = [
    c
    for c in list(data.columns)
    if "clf_" in c and "_skip" not in c and "_aux" not in c
]

train_idxs, valid_idxs = train_test_split([i for i in range(len(smiles_list))])

if reg_columns:
    reg = data[reg_columns]
    y_reg = np.array(reg)
    mdl = RegressionEstimator(
        n_outputs=y_reg.shape[1],
        fmap_shape1=X1.shape[1:],
        fmap_shape2=X2.shape[1:],
        dense_layers=[128, 64],
        batch_size=8,
        y_scale=None,
        patience=5,
        gpuid=GPUID,
    )
    mdl.fit(
        (X1[train_idxs], X2[train_idxs]),
        y_reg[train_idxs],
        (X1[valid_idxs], X2[valid_idxs]),
        y_reg[valid_idxs],
    )
    save_model(mdl, os.path.join(model_path, "reg"))

if clf_columns:
    clf = data[clf_columns]
    y_clf = np.array(clf)
    y_clf = to_categorical(y_clf)
    mdl = MultiClassEstimator(
        n_outputs=y_clf.shape[1],
        fmap_shape1=X1.shape[1:],
        fmap_shape2=X2.shape[1:],
        dense_layers=[128, 64],
        batch_size=8,
        patience=5,
        metric="ROC",
        gpuid=GPUID,
    )
    mdl.fit(
        (X1[train_idxs], X2[train_idxs]),
        y_clf[train_idxs],
        (X1[valid_idxs], X2[valid_idxs]),
        y_clf[valid_idxs],
    )
    save_model(mdl, os.path.join(model_path, "clf"))
