import numpy as np

from sklearn import metrics
from sklearn.metrics import auc, roc_curve, r2_score, mean_absolute_error

import matplotlib as plt
import seaborn as sns
import pandas as pd

from . import BasePlot
from .fetcher import ResultsFetcher
from stylia import NamedColors, NamedColorMaps, ContinuousColorMap


named_colors = NamedColors()
named_cmaps = NamedColorMaps()


class ActivesInactivesPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.is_available = True
            self.name = "actives-inactives"
            ax = self.ax

            rf = ResultsFetcher(path=path)
            y = rf.get_actives_inactives()
            actives = int(np.sum(y))
            inactives = len(y) - actives
            ax.bar(
                x=["Actives", "Inactives"],
                height=[actives, inactives],
                color=[named_colors.red, named_colors.blue],
            )
            direction = rf.get_direction()
            cut = rf.get_used_cut()
            ax.set_ylabel("Number of compounds")
            ax.set_xlabel(
                "Direction: {0}, Threshold: {1}".format(direction, round(cut, 3))
            )
            ax.set_title("Ratio actives:inactives")
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
            if len(set(bp)) > 1:
                class_names = ["I (0)", "A (1)"]
                disp = metrics.ConfusionMatrixDisplay(
                    metrics.confusion_matrix(bt, bp), display_labels=class_names
                )
                disp.plot(ax=ax, cmap=plt.cm.Greens, colorbar=False)
                # for labels in disp.text_.ravel():
                # labels.set_fontsize(22)
                ax.grid(False)
                ax.set_title("Confusion matrix")
            else:
                self.is_available = False
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
            ax.plot(fpr, tpr, color=named_colors.blue)
            ax.plot([0, 1], [0, 1], color=named_colors.gray)
            ax.set_title("AUROC = {0}".format(round(auc(fpr, tpr), 2)))
            ax.set_xlabel("1-Specificity (FPR)")
            ax.set_ylabel("Sensitivity (TPR)")
        else:
            self.is_available = False


class ScoreViolinPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.is_available = True
            self.name = "score-violin"
            ax = self.ax
            bt = ResultsFetcher(path=path).get_actives_inactives()
            yp = ResultsFetcher(path=path).get_pred_proba_clf()
            data = pd.DataFrame({"yp": yp, "bt": bt})
            sns.violinplot(
                x="bt",
                y="yp",
                data=data,
                ax=ax,
                palette=[named_colors.blue, named_colors.red],
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Inactive", "Active"])
            ax.set_title("Score distribution")
            ax.set_xlabel("")
            ax.set_ylabel("Classifier score (probability)")
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
            ax.scatter(x, y, color=named_colors.red)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_ylabel("AUROC")
            self.is_available = True
        else:
            self.is_available = False


class IndividualEstimatorsClassificationScorePlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.name = "raw-classification-scores"
            ax = self.ax
            self.fetcher = ResultsFetcher(path=path)
            tasks = self.fetcher.get_clf_tasks()
            task = tasks[0]
            df_ys = self.fetcher._read_individual_estimator_results(task)
            vals = []
            labels = []
            for yp in list(df_ys.columns):
                vals += list(df_ys[yp])
                labels += [yp] * len(df_ys[yp])
            data = pd.DataFrame({"label": labels, "values": vals})
            sns.boxplot(x="label", y="values", data=data, ax=ax)
            ax.set_ylabel("Classification score (probability)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_xlabel("")
            self.is_available = True
        else:
            self.is_available = False


class IndividualEstimatorsR2Plot(BasePlot):
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
            ax.scatter(x, y, color=named_colors.red)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90)
            ax.set_ylabel("R2")
            self.is_available = True
        else:
            self.is_available = False


class ProjectionUmapPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        self.name = "projection-umap"
        ax = self.ax
        if self.has_clf_data():
            if self.is_predict():
                bp = ResultsFetcher(path=path).get_actives_inactives_trained()
                bp_a = []
                bp_i = []
                for i, v in enumerate(bp):
                    if v == 1:
                        bp_a += [i]
                    if v == 0:
                        bp_i += [i]
                red0, red1 = ResultsFetcher(path=path).get_projections_umap_trained()
                ax.scatter(
                    [red0[i] for i in bp_i],
                    [red1[i] for i in bp_i],
                    color=named_colors.blue,
                    s=5,
                    label="Train known inactives",
                    edgecolors="none",
                )
                ax.scatter(
                    [red0[i] for i in bp_a],
                    [red1[i] for i in bp_a],
                    color=named_colors.red,
                    s=5,
                    label="Train known actives",
                    edgecolors="none",
                )
            bp = ResultsFetcher(path=path).get_actives_inactives()
            bp_a = []
            bp_i = []
            for i, v in enumerate(bp):
                if v == 1:
                    bp_a += [i]
                if v == 0:
                    bp_i += [i]
            red0, red1 = ResultsFetcher(path=path).get_projections_umap()
            ax.scatter(
                [red0[i] for i in bp_i],
                [red1[i] for i in bp_i],
                facecolor="none",
                edgecolors=named_colors.blue,
                s=15,
                label="Known inactives",
                zorder=1000000,
                lw=0.5,
            )
            ax.scatter(
                [red0[i] for i in bp_a],
                [red1[i] for i in bp_a],
                facecolor="none",
                edgecolors=named_colors.red,
                s=15,
                label="Known actives",
                zorder=100000000,
                lw=0.5,
            )
            y_pred = ResultsFetcher(path=path).get_pred_proba_clf()
            cmap = ContinuousColorMap(cmap=named_cmaps.coolwarm)
            # cmap.fit([0, 1])
            cmap.fit(y_pred)
            colors = cmap.transform(y_pred)
            ax.scatter(
                red0,
                red1,
                color=colors,
                alpha=0.7,
                s=15,
                zorder=100000,
                edgecolors="none",
            )
            self.is_available = True
            ax.set_title("UMAP 2D Projection")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.legend()
        else:
            self.is_available = False


class ProjectionPcaPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        self.name = "projection-pca"
        ax = self.ax
        if self.has_clf_data():
            if self.is_predict():
                bp = ResultsFetcher(path=path).get_actives_inactives_trained()
                bp_a = []
                bp_i = []
                for i, v in enumerate(bp):
                    if v == 1:
                        bp_a += [i]
                    if v == 0:
                        bp_i += [i]
                red0, red1 = ResultsFetcher(path=path).get_projections_pca_trained()
                ax.scatter(
                    [red0[i] for i in bp_i],
                    [red1[i] for i in bp_i],
                    color=named_colors.blue,
                    s=5,
                    label="Train known inactives",
                    edgecolors="none",
                )
                ax.scatter(
                    [red0[i] for i in bp_a],
                    [red1[i] for i in bp_a],
                    color=named_colors.red,
                    s=5,
                    label="Train known actives",
                    edgecolors="none",
                )
            bp = ResultsFetcher(path=path).get_actives_inactives()
            bp_a = []
            bp_i = []
            for i, v in enumerate(bp):
                if v == 1:
                    bp_a += [i]
                if v == 0:
                    bp_i += [i]
            red0, red1 = ResultsFetcher(path=path).get_projections_pca()
            ax.scatter(
                [red0[i] for i in bp_i],
                [red1[i] for i in bp_i],
                facecolor="none",
                edgecolors=named_colors.blue,
                s=15,
                label="Known inactives",
                zorder=1000000,
                lw=0.5,
            )
            ax.scatter(
                [red0[i] for i in bp_a],
                [red1[i] for i in bp_a],
                facecolor="none",
                edgecolors=named_colors.red,
                s=15,
                label="Known actives",
                zorder=100000000,
                lw=0.5,
            )
            y_pred = ResultsFetcher(path=path).get_pred_proba_clf()
            cmap = ContinuousColorMap(cmap=named_cmaps.coolwarm)
            # cmap.fit([0, 1])
            cmap.fit(y_pred)
            colors = cmap.transform(y_pred)
            ax.scatter(
                red0,
                red1,
                color=colors,
                alpha=0.7,
                s=15,
                zorder=100000,
                edgecolors="none",
            )
            self.is_available = True
            ax.set_title("PCA 2D Projection")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.legend()
        else:
            self.is_available = False


class RegressionPlotTransf(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_reg_data():
            self.is_available = True
            self.name = "regression-trans"
            ax = self.ax
            yt = ResultsFetcher(path=path).get_transformed()
            yp = ResultsFetcher(path=path).get_pred_reg_trans()
            ax.scatter(yt, yp, color=named_colors.purple, s=15, alpha=0.7)
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
            ax.hist(yp, color=named_colors.green)
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
            print(len(yt))
            print(len(yp))
            ax.scatter(yt, yp, color=named_colors.green, s=15, alpha=0.7)
            ax.set_xlabel("Observed Activity")
            ax.set_ylabel("Predicted Activity")
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
            ax.hist(yp, color=named_colors.green)
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
            ax.scatter(yt, ytrans, color=named_colors.green, s=15, alpha=0.7)
            ax.set_xlabel("Observed Activity (Raw)")
            ax.set_ylabel("Observed Activity (Transformed)")
            ax.set_title("Continuous data transformation")
        else:
            self.is_available = False
