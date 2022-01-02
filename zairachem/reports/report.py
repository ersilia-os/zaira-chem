
from .plots import ActivesInactivesPlot

from .. import ZairaBase
from ..vars import REPORT_SUBFOLDER


class Reporter(ZairaBase):

    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _actives_inactives_plot(self):
        ActivesInactivesPlot(ax=None, path=self.path).save()

    def run(self):
        self._actives_inactives_plot()