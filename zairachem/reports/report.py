from .plots import (
    ActivesInactivesPlot,
    ConfusionPlot,
    RocCurvePlot,
    ProjectionPlot,
    RegressionPlotRaw,
    HistogramPlotRaw,
    RegressionPlotTransf,
    HistogramPlotTransf,
    Transformation,
)

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

    def _confusion_matrix_plot(self):
        ConfusionPlot(ax=None, path=self.path).save()

    def _roc_curve_plot(self):
        RocCurvePlot(ax=None, path=self.path).save()

    def _projection_plot(self):
        ProjectionPlot(ax=None, path=self.path).save()

    def _regression_plot_raw(self):
        RegressionPlotRaw(ax=None, path=self.path).save()

    def _histogram_plot_raw(self):
        HistogramPlotRaw(ax=None, path=self.path).save()

    def _regression_plot_transf(self):
        RegressionPlotTransf(ax=None, path=self.path).save()

    def _histogram_plot_transf(self):
        HistogramPlotTransf(ax=None, path=self.path).save()

    def _transformation_plot(self):
        Transformation(ax=None, path=self.path).save()

    def run(self):
        self._actives_inactives_plot()
        self._confusion_matrix_plot()
        self._roc_curve_plot()
        self._projection_plot()
        self._regression_plot_transf()
        self._histogram_plot_transf()
        self._regression_plot_raw()
        self._histogram_plot_raw()
        self._transformation_plot()
