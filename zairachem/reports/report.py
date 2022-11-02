import os

from .table import OutputTable, PerformanceTable

from .plots import (
    ActivesInactivesPlot,
    ConfusionPlot,
    IndividualEstimatorsClassificationScorePlot,
    RocCurvePlot,
    ScoreViolinPlot,
    ScoreStripPlot,
    ProjectionUmapPlot,
    ProjectionPcaPlot,
    RegressionPlotRaw,
    HistogramPlotRaw,
    RegressionPlotTransf,
    HistogramPlotTransf,
    Transformation,
    IndividualEstimatorsAurocPlot,
    IndividualEstimatorsR2Plot,
    TanimotoSimilarityToTrainPlot,
)

from .. import ZairaBase


class Reporter(ZairaBase):
    def __init__(self, path, plot_name=None):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)
        self.plot_name = plot_name

    def __skip(self, this_name):
        if self.plot_name is None:
            return False
        if self.plot_name == this_name:
            return False
        else:
            return True

    def _actives_inactives_plot(self):
        if not self.__skip("actives-inactives"):
            ActivesInactivesPlot(ax=None, path=self.path).save()

    def _confusion_matrix_plot(self):
        if not self.__skip("confusion-matrix"):
            ConfusionPlot(ax=None, path=self.path).save()

    def _roc_curve_plot(self):
        if not self.__skip("roc-curve"):
            RocCurvePlot(ax=None, path=self.path).save()

    def _score_violin_plot(self):
        if not self.__skip("score-violin"):
            ScoreViolinPlot(ax=None, path=self.path).save()

    def _score_strip_plot(self):
        if not self.__skip("score-strip"):
            ScoreStripPlot(ax=None, path=self.path).save()

    def _projection_umap_plot(self):
        if not self.__skip("projection-umap"):
            ProjectionUmapPlot(ax=None, path=self.path).save()

    def _projection_pca_plot(self):
        if not self.__skip("projection-pca"):
            ProjectionPcaPlot(ax=None, path=self.path).save()

    def _regression_plot_raw(self):
        if not self.__skip("regression-raw"):
            RegressionPlotRaw(ax=None, path=self.path).save()

    def _histogram_plot_raw(self):
        if not self.__skip("histogram-raw"):
            HistogramPlotRaw(ax=None, path=self.path).save()

    def _regression_plot_transf(self):
        if not self.__skip("regression-trans"):
            RegressionPlotTransf(ax=None, path=self.path).save()

    def _histogram_plot_transf(self):
        if not self.__skip("histogram-trans"):
            HistogramPlotTransf(ax=None, path=self.path).save()

    def _transformation_plot(self):
        if not self.__skip("transformation"):
            Transformation(ax=None, path=self.path).save()

    def _individual_estimators_auroc_plot(self):
        if not self.__skip("roc-individual"):
            IndividualEstimatorsAurocPlot(ax=None, path=self.path).save()

    def _individual_estimators_classification_score_plot(self):
        if not self.__skip("raw-classification-scores"):
            IndividualEstimatorsClassificationScorePlot(ax=None, path=self.path).save()

    def _individual_estimators_r2_plot(self):
        if not self.__skip("r2-individual"):
            IndividualEstimatorsR2Plot(ax=None, path=self.path).save()

    def _tanimoto_similarity_to_train_plot(self):
        if not self.__skip("tanimoto-similarity-to-train"):
            TanimotoSimilarityToTrainPlot(ax=None, path=self.path).save()

    def _output_table(self):
        OutputTable(path=self.path).run()

    def _performance_table(self):
        PerformanceTable(path=self.path).run()

    def run(self):
        self._output_table()
        self._performance_table()
        self._actives_inactives_plot()
        self._confusion_matrix_plot()
        self._roc_curve_plot()
        self._score_violin_plot()
        self._score_strip_plot()
        self._projection_umap_plot()
        self._projection_pca_plot()
        self._regression_plot_transf()
        self._histogram_plot_transf()
        self._regression_plot_raw()
        self._histogram_plot_raw()
        self._transformation_plot()
        self._individual_estimators_auroc_plot()
        self._individual_estimators_classification_score_plot()
        self._individual_estimators_r2_plot()
        self._tanimoto_similarity_to_train_plot()
