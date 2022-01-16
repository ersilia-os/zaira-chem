#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from FCD import get_predictions, calculate_frechet_distance

# Load molecules to compare
mols1 = pd.read_csv(sys.argv[1], header=None)[0].values[:5000]  # take at least 5000 molecules
mols2 = pd.read_csv(sys.argv[2], header=None)[0].values[:5000]  # take at least 5000 molecules

# get ChEMBLNET activations of generated molecules
mol_act1 = get_predictions(mols1)
mol_act2 = get_predictions(mols2)
FCD = calculate_frechet_distance(mu1=np.mean(mol_act1, axis=0), mu2=np.mean(mol_act2, axis=0),
                                 sigma1=np.cov(mol_act1.T), sigma2=np.cov(mol_act2.T))
print("FCD: %.4f" % FCD)
