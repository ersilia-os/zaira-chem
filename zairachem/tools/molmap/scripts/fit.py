import sys
import os

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "../bidd-molmap/"))

import pandas as pd
import numpy as np

from molmap.model import save_model
from molmap.model import RegressionEstimator, MultiClassEstimator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

file_name = sys.argv[1]
x1_path = sys.argv[2]
x2_path = sys.argv[3]
model_path = sys.argv[4]

SMILES_COLUMN = "smiles"
GPUID = 1

data = pd.read_csv(file_name)

smiles_list = list(data[SMILES_COLUMN])

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

with open(x1_path, "rb") as f:
    X1 = np.load(f)

with open(x2_path, "rb") as f:
    X2 = np.load(f)

X1_outer_shape = X1.shape[1:]
X2_outer_shape = X2.shape[1:]

if reg_columns:
    reg = data[reg_columns]
    y_reg = np.array(reg)
    y_reg_train = y_reg[train_idxs]
    y_reg_valid = y_reg[valid_idxs]
    mdl = RegressionEstimator(
        n_outputs=y_reg.shape[1],
        fmap_shape1=X1_outer_shape,
        fmap_shape2=X2_outer_shape,
        dense_layers=[128, 64],
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
