import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import roc_auc_score, r2_score

TEST_PROP = 0.2
N_SPLITS = 3

RANDOM_STATE = None

def validation_score(mod, X, y, is_clf):

    if is_clf:
        spl = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_PROP, random_state=RANDOM_STATE)
        metric = roc_auc_score
    else:
        spl = ShuffleSplit(n_splits=N_SPLITS, test_size=TEST_PROP, random_state=RANDOM_STATE)
        metric = r2_score

    score = []
    for train_idx, test_idx in spl.split(X=X, y=y):
        mod.fit(X[train_idx], y[train_idx])
        if is_clf:
            y_pred = mod.predict_proba(X[test_idx])[:,1]
        else:
            y_pred = mod.predict(X[test_idx])
        y_true = y[test_idx]
        score += [metric(y_true, y_pred)]
    return np.mean(score)
