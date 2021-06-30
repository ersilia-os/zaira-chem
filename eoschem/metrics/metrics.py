from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score


class Metric(object):
    def __init__(self, is_clf):
        self.is_clf = is_clf

    def _clf_score(self, y_true, y_pred):
        metric = "roc_auc_score"
        score = roc_auc_score(y_true, y_pred[:, 1])  #  TODO: Adapt for multioutput
        return {"metric": metric, "score": score}

    def _reg_score(self, y_true, y_pred):
        metric = "r2_score"
        score = r2_score(y_true, y_pred)
        return {"metric": metric, "score": score}

    def score(self, y_true, y_pred):
        if self.is_clf:
            return self._clf_score(y_true, y_pred)
        else:
            return self._reg_score(y_true, y_pred)
