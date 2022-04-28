import os

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
    IndividualEstimatorsAurocPlot,
    IndividualEstimatorsR2Plot,
)

from .. import ZairaBase


class Reporter(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)

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

    def _individual_estimators_auroc_plot(self):
        IndividualEstimatorsAurocPlot(ax=None, path=self.path).save()

    def _individual_estimators_r2_plot(self):
        IndividualEstimatorsR2Plot(ax=None, path=self.path).save()

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
        self._individual_estimators_auroc_plot()
        self._individual_estimators_r2_plot()
