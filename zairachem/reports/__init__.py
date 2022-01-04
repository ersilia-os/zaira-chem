import matplotlib.pyplot as plt

import os
from zairachem.vars import REPORT_SUBFOLDER

from .. import ZairaBase

from .utils import set_style

set_style()


INDIVIDUAL_FIGSIZE = (5, 5)


class BasePlot(ZairaBase):
    def __init__(self, ax, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=INDIVIDUAL_FIGSIZE)
        self.name = "base"
        self.ax = ax

    def save(self):
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.path, REPORT_SUBFOLDER, self.name + ".png"), dpi=300
        )

    def load(self):
        pass
