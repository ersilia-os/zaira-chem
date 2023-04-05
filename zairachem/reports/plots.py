import numpy as np

from sklearn import metrics
from sklearn.metrics import auc, roc_curve, r2_score, mean_absolute_error

import matplotlib as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pandas as pd

from . import BasePlot
from .fetcher import ResultsFetcher
from stylia import NamedColors, NamedColorMaps, ContinuousColorMap


named_colors = NamedColors()
named_cmaps = NamedColorMaps()


class ActivesInactivesPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 5))
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
            y_min = 0
            y_max = max(actives, inactives)
            range = y_max - y_min
            ax.set_ylim(0 - range * 0.02, y_max + range * 0.1)
            ax.text(
                0,
                actives + y_max * 0.02,
                actives,
                va="center",
                ha="center",
                color=named_colors.red,
            )
            ax.text(
                1,
                inactives + y_max * 0.02,
                inactives,
                va="center",
                ha="center",
                color=named_colors.blue,
            )
            direction = rf.get_direction()
            cut = rf.get_used_cut()
            ax.set_ylabel("Number of compounds")
            if cut == 1:
                ax.set_xlabel("")
            else:
                ax.set_xlabel(
                    "Activity direction: {0}, Threshold: {1}".format(
                        direction, round(cut, 3)
                    )
                )
            p = np.round(actives / len(y) * 100, 1)
            q = np.round(100 - p, 1)
            ax.set_title("Actives = {0}%, Inactives = {1}%".format(p, q))
        else:
            self.is_available = False


class ConfusionPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if self.has_clf_data():
            self.is_available = True
            self.name = "confusion-matrix"
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
        BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 3))
        if self.has_clf_data():
            self.is_available = True
            self.name = "roc-curve"
            ax = self.ax
            cmap = ContinuousColorMap(cmap="spectral")
            cmap.fit([0, 1])
            bt = ResultsFetcher(path=path).get_actives_inactives()
            yp = ResultsFetcher(path=path).get_pred_proba_clf()
            fpr, tpr, _ = roc_curve(bt, yp)
            auroc = auc(fpr, tpr)
            color = cmap.transform([auroc])[0]
            ax.plot(fpr, tpr, color=color, zorder=10000, lw=1)
            ax.fill_between(fpr, tpr, color=color, alpha=0.5, lw=0, zorder=1000)
            ax.plot([0, 1], [0, 1], color=named_colors.gray, lw=1)
            ax.set_title("AUROC = {0}".format(round(auroc, 2)))
            ax.set_xlabel("1-Specificity (FPR)")
            ax.set_ylabel("Sensitivity (TPR)")
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        else:
            self.is_available = False


class ScoreViolinPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 5))
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


class ScoreStripPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path, figsize=(3, 5))
        self.MAX_SAMPLES = 1000
        if self.has_clf_data():
            self.is_available = True
            self.name = "score-strip"
            ax = self.ax
            bt = ResultsFetcher(path=path).get_actives_inactives()
            yp = ResultsFetcher(path=path).get_pred_proba_clf()
            data = pd.DataFrame({"yp": yp, "bt": bt})
            data_a = data[data["bt"] == 1]
            data_i = data[data["bt"] == 0]
            if data_a.shape[0] > self.MAX_SAMPLES:
                data_a = data_a.sample(n=self.MAX_SAMPLES)
            if data_i.shape[0] > self.MAX_SAMPLES:
                data_i = data_i.sample(n=self.MAX_SAMPLES)
            y_i = data_i["yp"]
            y_a = data_a["yp"]
            n_i = np.random.uniform(-0.3, 0.3, len(y_i))
            n_a = np.random.uniform(-0.3, 0.3, len(y_a))
            ax.scatter(n_a + 1, y_a, color=named_colors.red, zorder=1, alpha=0.5, s=20)
            ax.scatter(n_i, y_i, color=named_colors.blue, zorder=1, alpha=0.5, s=20)
            p05 = np.percentile(data_a["yp"], 5)
            p25 = np.percentile(data_a["yp"], 25)
            p50 = np.percentile(data_a["yp"], 50)
            p75 = np.percentile(data_a["yp"], 75)
            p95 = np.percentile(data_a["yp"], 95)
            r = Rectangle(
                (0.85, p25),
                0.3,
                p75 - p25,
                color=named_colors.red,
                alpha=0.5,
                zorder=20000,
                lw=0,
                edgecolor=named_colors.red,
            )
            ax.plot([1, 1], [p75, p95], lw=1, color=named_colors.black, zorder=20000)
            ax.plot([1, 1], [p05, p25], lw=1, color=named_colors.black, zorder=20000)
            ax.plot(
                [0.85, 1.15], [p25, p25], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [0.85, 1.15], [p50, p50], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [0.85, 1.15], [p75, p75], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [0.85, 0.85], [p25, p75], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [1.15, 1.15], [p25, p75], lw=1, color=named_colors.black, zorder=20000
            )
            ax.add_patch(r)
            p05 = np.percentile(data_i["yp"], 5)
            p25 = np.percentile(data_i["yp"], 25)
            p50 = np.percentile(data_i["yp"], 50)
            p75 = np.percentile(data_i["yp"], 75)
            p95 = np.percentile(data_i["yp"], 95)
            r = Rectangle(
                (-0.15, p25),
                0.3,
                p75 - p25,
                color=named_colors.blue,
                alpha=0.5,
                zorder=20000,
                lw=0,
                edgecolor=named_colors.blue,
            )
            ax.plot([0, 0], [p75, p95], lw=1, color=named_colors.black, zorder=20000)
            ax.plot([0, 0], [p05, p25], lw=1, color=named_colors.black, zorder=20000)
            ax.plot(
                [-0.15, 0.15], [p25, p25], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [-0.15, 0.15], [p50, p50], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [-0.15, 0.15], [p75, p75], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [-0.15, -0.15], [p25, p75], lw=1, color=named_colors.black, zorder=20000
            )
            ax.plot(
                [0.15, 0.15], [p25, p75], lw=1, color=named_colors.black, zorder=20000
            )
            ax.add_patch(r)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Inactive", "Active"])
            ax.set_title("Score distribution")
            ax.set_xlabel("")
            ax.set_ylabel("Classifier score (probability)")
            ax.set_xlim(-0.5, 1.5)

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
            for yp in sorted(df_ys.columns):
                fpr, tpr, _ = roc_curve(bt, list(df_ys[yp]))
                aucs += [auc(fpr, tpr)]
                labels += [yp]
            y = [i for i in range(len(labels))]
            x = aucs
            cmap = ContinuousColorMap("spectral")
            cmap.fit([0.5, 1])
            colors = cmap.transform(x)

            def format(l):
                l = l.replace("_", " ").replace("-", " ")
                return l.title()

            for i in y:
                ax.text(
                    0.75,
                    i,
                    "{0} / {1}".format(format(labels[i]), np.round(x[i], 3)),
                    va="center",
                    ha="center",
                )
            for i in y:
                r = Rectangle((0, i - 0.3), x[i], 0.6, color=colors[i], alpha=0.5)
                ax.add_patch(r)
            ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_ylabel("Estimators")
            ax.set_yticklabels("")
            ax.set_xlim(0.45, 1.05)
            ax.set_ylim(-0.6, len(labels) - 0.4)
            ax.set_yticks(y)
            ax.set_xlabel("AUROC")
            ax.set_title("Individual performances")
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
            if (
                self.is_predict()
                and ResultsFetcher(path=path).get_projections_umap_trained() is not None
            ):
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
            if (
                self.is_predict()
                and ResultsFetcher(path=path).get_projections_pca_trained() is not None
            ):
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


class TanimotoSimilarityToTrainPlot(BasePlot):
    def __init__(self, ax, path):
        BasePlot.__init__(self, ax=ax, path=path)
        if (
            ResultsFetcher(path=path).get_tanimoto_similarities_to_training_set()
            is not None
        ):
            self.name = "tanimoto-similarity-to-train"
            ax = self.ax
            df = ResultsFetcher(path=path).get_tanimoto_similarities_to_training_set()
            columns = [c for c in list(df.columns) if c.startswith("sim")]
            df = df[columns]
            cmap = ContinuousColorMap(cmap=named_cmaps.spectral)
            cmap.fit([i for i in range(len(columns))])
            colors = cmap.transform([i for i in range(len(columns))])
            for i, col in enumerate(columns):
                ax.hist(list(df[col]), cumulative=True, color=colors[i])
            ax.set_xlabel("Tanimoto similarity")
            ax.set_ylabel("Cumulative proportion")
            ax.set_title("Tanimoto similarity to train")
            self.is_available = True
        else:
            self.is_available = False
