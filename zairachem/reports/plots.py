import numpy as np

from sklearn import metrics
from sklearn.metrics import auc, roc_curve, r2_score, mean_absolute_error
from scipy.stats import gaussian_kde


import matplotlib as plt
from . import BasePlot
from .fetcher import ResultsFetcher
from .utils import ersilia_colors


class ActivesInactivesPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.is_available = True
            self.name = "actives-inactives"
            ax = self.ax

            y = ResultsFetcher(path=path).get_actives_inactives()
            actives = int(np.sum(y))
            inactives = len(y) - actives
            ax.bar(
                x=["Actives", "Inactives"],
                height=[actives, inactives],
                color=[ersilia_colors["pink"], ersilia_colors["blue"]],
            )
            ax.set_ylabel("Number of compounds")
        else:
            self.is_available = False


class ConfusionPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.is_available = True
            self.name = "contingency"
            ax = self.ax

            bt = ResultsFetcher(path=path).get_actives_inactives()
            bp = ResultsFetcher(path=path).get_pred_binary_clf()
            class_names = ["I (0)", "A (1)"]
            disp = metrics.ConfusionMatrixDisplay(
                metrics.confusion_matrix(bt, bp), display_labels=class_names
            )
            disp.plot(ax=ax, cmap=plt.cm.Greens, colorbar=False)
            for labels in disp.text_.ravel():
                labels.set_fontsize(22)
            ax.grid(False)
            ax.set_title("Confusion matrix")
        else:
            self.is_available = False


class RocCurvePlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.is_available = True
            self.name = "roc-curve"
            ax = self.ax
            bt = ResultsFetcher(path=path).get_actives_inactives()
            yp = ResultsFetcher(path=path).get_pred_proba_clf()
            fpr, tpr, _ = roc_curve(bt, yp)
            ax.plot(fpr, tpr, color=ersilia_colors["mint"])
            ax.set_title("ROC AUC {0}".format(round(auc(fpr, tpr), 3)))
            ax.set_title("AUROC = {0}".format(round(auc(fpr, tpr), 2)))
            ax.set_xlabel("1-Specificity (FPR)")
            ax.set_ylabel("Sensitivity (TPR)")
        else:
            self.is_available = False


class IndividualEstimatorsAurocPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.name = "roc-individual"
            ax = self.ax
            self.fetcher = ResultsFetcher(path=path)
            tasks = self.fetcher.get_clf_tasks()
            task = tasks[0]
            bt = self.fetcher.get_actives_inactives()
            df_ys = self.fetcher._read_individual_estimator_results(task)
            aucs = []
            labels = []
            for yp in list(df_ys.columns):
                fpr, tpr, _ = roc_curve(bt, list(df_ys[yp]))
                aucs += [auc(fpr, tpr)]
                labels += [yp]
            x = [i for i in range(len(labels))]
            y = aucs
            ax.scatter(x, y)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_ylabel("AUROC")
            self.is_available = True
        else:
            self.is_available = False


class InidvidualEstimatorsR2Plot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_reg_data():
            self.name = "r2-individual"
            ax = self.ax
            self.fetcher = ResultsFetcher(path=path)
            tasks = self.fetcher.get_reg_tasks()
            task = tasks[0]
            yt = ResultsFetcher(path=path).get_transformed()
            df_ys = self.fetcher._read_individual_estimator_results(task)
            scores = []
            labels = []
            for yp in list(df_ys.columns):
                scores += [r2_score(yt, list(df_ys[yp]))]
                labels += [yp]
            x = [i for i in range(len(labels))]
            y = scores
            ax.scatter(x, y)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_ylabel("R2")
            self.is_available = True
        else:
            self.is_available = False

class ProjectionPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        self.name = "projection"
        ax = self.ax
        bp = ResultsFetcher(path=path).get_pred_binary_clf()
        bp_a = []
        bp_i = []
        for i, v in enumerate(bp):
            if v == 1:
                bp_a += [i]
            if v == 0:
                bp_i += [i]
        umap0, umap1 = ResultsFetcher(path=path).get_projections()
        if self.is_predict():
            umap0_tr, umap1_tr = ResultsFetcher(path=path).get_projections_trained()
            ax.scatter(umap0_tr, umap1_tr, color="gray", s=5)
        ax.scatter(
            [umap0[i] for i in bp_i],
            [umap1[i] for i in bp_i],
            color=ersilia_colors["blue"],
            alpha=0.7,
            s=15,
        )
        ax.scatter(
            [umap0[i] for i in bp_a],
            [umap1[i] for i in bp_a],
            color=ersilia_colors["pink"],
            alpha=0.7,
            s=15,
        )
        self.is_available = True


class RegressionPlotTransf(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_reg_data():
            self.is_available = True
            self.name = "regression-trans"
            ax = self.ax
            yt = ResultsFetcher(path=path).get_transformed()
            yp = ResultsFetcher(path=path).get_pred_reg_trans()
            ax.scatter(yt, yp, c=ersilia_colors["dark"], s=15, alpha=0.7)
            ax.set_xlabel("Observed Activity (Transformed)")
            ax.set_ylabel("Predicted Activity (Transformed)")
            ax.set_title(
                "R2 = {0} | MAE = {1}".format(
                    round(r2_score(yt, yp), 3), round(mean_absolute_error(yt, yp), 3)
                )
            )
        else:
            self.is_available = False


class HistogramPlotTransf(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_reg_data():
            self.is_available = True
            self.name = "histogram-trans"
            ax = self.ax
            yp = ResultsFetcher(path=path).get_pred_reg_trans()
            ax.hist(yp, color=ersilia_colors["mint"])
            ax.set_xlabel("Predicted Activity")
            ax.set_ylabel("Frequency")
            ax.set_title("Predicted activity distribution")
        else:
            self.is_available = False


class RegressionPlotRaw(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_reg_data():
            self.is_available = True
            self.name = "regression-raw"
            ax = self.ax
            yt = ResultsFetcher(path=path).get_raw()
            yp = ResultsFetcher(path=path).get_pred_reg_raw()
            ax.scatter(yt, yp, c=ersilia_colors["dark"], s=15, alpha=0.7)
            ax.set_xlabel("Observed Activity (Transformed)")
            ax.set_ylabel("Predicted Activity (Transformed)")
            ax.set_title(
                "R2 = {0} | MAE = {1}".format(
                    round(r2_score(yt, yp), 3), round(mean_absolute_error(yt, yp), 3)
                )
            )
        else:
            self.is_available = False


class HistogramPlotRaw(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_reg_data():
            self.is_available = True
            self.name = "histogram-raw"
            ax = self.ax
            yp = ResultsFetcher(path=path).get_pred_reg_raw()
            ax.hist(yp, color=ersilia_colors["mint"])
            ax.set_xlabel("Predicted Activity")
            ax.set_ylabel("Frequency")
            ax.set_title("Predicted activity distribution")
        else:
            self.is_available = False


class Transformation(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_reg_data():
            self.is_available = True
            self.name = "transformation"
            ax = self.ax
            yt = ResultsFetcher(path=path).get_raw()
            ytrans = ResultsFetcher(path=path).get_transformed()
            ax.scatter(yt, ytrans, c=ersilia_colors["dark"], s=15, alpha=0.7)
            ax.set_xlabel("Observed Activity (Raw)")
            ax.set_ylabel("Observed Activity (Transformed)")
        else:
            self.is_available = False
