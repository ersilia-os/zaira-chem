import matplotlib.pyplot as plt

import os
import pandas as pd
from zairachem.vars import DATA_FILENAME, DATA_SUBFOLDER, REPORT_SUBFOLDER
import stylia

from .. import ZairaBase


INDIVIDUAL_FIGSIZE = (5, 5)


class BaseResults(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def has_outcome_data(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        for c in list(df.columns):
            if "clf_" in c:
                return True
            if "reg_" in c:
                return True
        return False

    def has_clf_data(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        for c in list(df.columns):
            if "clf_" in c and "_skip" not in c and "_aux" not in c:
                return True
        return False

    def has_reg_data(self):
        df = pd.read_csv(os.path.join(self.path, DATA_SUBFOLDER, DATA_FILENAME))
        for c in list(df.columns):
            if "reg_" in c and "_skip" not in c and "_aux" not in c:
                return True
        return False


class BaseTable(BaseResults):
    def __init__(self, path):
        BaseResults.__init__(self, path=path)


class BasePlot(BaseResults):
    def __init__(self, ax, path, figsize=None):
        BaseResults.__init__(self, path=path)
        if ax is None:
            if figsize is None:
                figsize = INDIVIDUAL_FIGSIZE
            _, ax = stylia.create_figure(1, 1, figsize=figsize)
        self.name = "base"
        self.ax = ax[0]

    def save(self):
        if self.is_available:
            stylia.save_figure(
                os.path.join(self.path, REPORT_SUBFOLDER, self.name + ".png")
            )

    def load(self):
        pass
