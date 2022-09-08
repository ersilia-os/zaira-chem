import os
import shutil
import numpy as np
import pandas as pd
from ..tools.autogluon.multilabel import MultilabelPredictor, TabularDataset

AUTOGLUON_TIME_BUDGET_SECONDS = 300
AUTOGLUON_MINIMUM_TIME_BUDGET_SECONDS = 60
AUTOGLUON_MAXIMUM_TIME_BUDGET_SECONDS = 600


class AutoGluonEstimator(object):
    def __init__(self, save_path, time_budget=AUTOGLUON_TIME_BUDGET_SECONDS):
        self.save_path = os.path.abspath(save_path)
        if time_budget is None:
            time_budget = 0
        time_budget = max(time_budget, AUTOGLUON_MINIMUM_TIME_BUDGET_SECONDS)
        time_budget = min(time_budget, AUTOGLUON_MAXIMUM_TIME_BUDGET_SECONDS)
        self.time_limit = int(time_budget)
        self.model = None

    def _fit(self, data, labels, groups):
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path)
        df = TabularDataset(data)
        problem_types = []
        eval_metrics = []
        for l in labels:
            if "clf" in l:
                problem_types += ["binary"]
                eval_metrics += ["roc_auc"]
            else:
                problem_types += ["regression"]
                eval_metrics += ["r2"]
        self.model = MultilabelPredictor(
            labels=labels,
            path=self.save_path,
            problem_types=problem_types,
            eval_metrics=eval_metrics,
            consider_labels_correlation=False,
            groups=groups,
        )
        self.model.fit(
            train_data=df,
            time_limit=self.time_limit,
            refit_full=True,
            # presets='high_quality_fast_inference_only_refit',
        )

    def get_out_of_sample(self):
        O = []
        labels = []
        for label in self.model.labels:
            estimator = self.model.get_predictor(label)
            if "clf" in label:
                O += [list(estimator.get_oof_pred_proba()[1])]
                O += [list(estimator.get_oof_pred())]
                labels += [label, label + "_bin"]
            else:
                O += [list(estimator.get_oof_pred())]
                labels += [label]
        O = np.array(O).T
        df = pd.DataFrame(O, columns=labels)
        return df

    def fit(self, data, labels, groups):
        self._fit(data=data, labels=labels, groups=groups)

    def fit_predict(self, data, labels, groups):
        self.fit(data=data, labels=labels, groups=groups)
        df = self.get_out_of_sample()
        return df

    def save(self):
        self.model.save()

    def load(self):
        model = MultilabelPredictor.load(self.save_path)
        return AutoGluonEstimatorArtifact(model)


class AutoGluonEstimatorArtifact(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        P = []
        labels = []
        for label in self.model.labels:
            estimator = self.model.get_predictor(label)
            if "clf" in label:
                P += [list(estimator.predict_proba(data)[1])]
                P += [list(estimator.predict(data))]
                labels += [label, label + "_bin"]
            else:
                P += [list(estimator.predict(data))]
                labels += [label]
        P = np.array(P).T
        df = pd.DataFrame(P, columns=labels)
        return df

    def run(self, data):
        results = self.predict(data)
        return results
