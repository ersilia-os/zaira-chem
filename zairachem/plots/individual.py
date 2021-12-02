import json
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy.stats import gaussian_kde
from sklearn import metrics

import seaborn as sns
from . import BasePlot


class ActivesInactivesPlot(BasePlot):

    def __init__(self, ax):
        BasePlot.__init__(self, ax=ax)
        self.name = "actives-inactives"
        ax.bar(x=["Actives", "Inactives"], height=[actives, inactives], color=["red", "blue"])
        ax.set_ylabel("Number of compounds")


class ConfusionPlot(BasePlot):

    def __init__(self, ax):
        BasePlot.__init__(self, ax=ax)
        self.name = "contingency-plot"
        class_names = ['I (0)', 'A (1)']
        is_train = len(clf.keys())>1
        bt = []
        bp = []
        for k,v in clf.items():
            if is_train and k == "main": continue
            bt += list(v["y_true"])
            bp += list(v["b_pred"])
        disp = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(bt, bp), display_labels = class_names)
        disp.plot(ax=ax, cmap=plt.cm.Greens, colorbar=False,)
        for labels in disp.text_.ravel():
            labels.set_fontsize(22)
        ax.grid(False)
        ax.set_title("Confusion matrix")


class ProjectionPlot(BasePlot):

    def __init__(self, ax):
        BasePlot.__init__(self, ax=ax)
        self.name = "projection-plot"
        ax.scatter(X_r[:,0], X_r[:,1])


class RegressionPlot(BasePlot):

    def __init__(self, ax):
        BasePlot.__init__(self, ax=ax)
        self.name = "regression-plot"
        is_train = len(reg.keys())>1
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        for k,v in reg.items():
            if is_train and k == "main": continue
            x, y = np.array(v["y_true"]), np.array(v["y_pred"])
            xy = np.vstack([x,y])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax.scatter(x, y, c=z, s=5, cmap="coolwarm")
        ax.set_xlabel("Observed Activity (Gaussianized)")
        ax.set_ylabel("Predicted Activity (Gaussianized)")
        ax.set_title("R2 = {0} | MAE = {0}".format("NULL", "NULL"))


class RocCurvePlot(BasePlot):

    def __init__(self, ax):
        BasePlot.__init__(self, ax=ax)
        self.name = "actives-inactives"
        is_train = len(clf.keys())>1
        for k,v in clf.items():
            if is_train and k == "main": continue
            fpr, tpr, _ = roc_curve(v["y_true"], v["y_pred"])
            ax.plot(fpr, tpr, color=blue, lw=2)
            break
        ax.set_title("AUROC = {0}".format(round(auc(fpr, tpr), 2)))
        ax.set_xlabel("1-Specificity (FPR)")
        ax.set_ylabel("Sensitivity (TPR)")
