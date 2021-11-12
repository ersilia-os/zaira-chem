import selfies as sf
import autokeras as ak
import numpy as np


class SmilesTextRegressor(object):

    def __init__(self):
        self.mdl = ak.TextRegressor(overwrite=True, max_trials=10)

    def tokenize(self, smiles_list):
        selfies = []
        for smi in smiles_list:
            selfies += [" ".join(sf.split_selfies(sf.encoder(smi)))]
        return np.array(selfies)

    def fit(self, smiles_list, y, epochs=100):
        y = np.array(y)
        X = self.tokenize(smiles_list)
        self.mdl.fit(X, y, validation_data=(X, y), epochs=epochs)

    def predict(self, smiles_list):
        X = self.tokenize(smiles_list)
        return self.mdl.predict(X)

    def save(self):
        pass

    def load(self):
        pass
