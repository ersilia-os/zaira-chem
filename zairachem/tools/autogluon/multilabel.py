from autogluon.tabular import TabularDataset, TabularPredictor

try:
    from autogluon.core.utils.utils import setup_outputdir
except:
    from autogluon.common.utils.utils import setup_outputdir
try:
    from autogluon.core.utils.loaders import load_pkl
except:
    from autogluon.common.utils.loaders import load_pkl
try:
    from autogluon.core.utils.savers import save_pkl
except:
    from autogluon.common.utils.savers import save_pkl
import os.path

from ... import ZairaBase
from ...vars import ESTIMATORS_SUBFOLDER, POOL_SUBFOLDER


def get_model_dir():
    zb = ZairaBase()
    model_dir = zb.get_trained_dir()
    return model_dir


class MultilabelPredictor(object):
    """Tabular Predictor for predicting multiple columns in table.
    Creates multiple TabularPredictor objects which you can also use individually.
    You can access the TabularPredictor for a particular label via: `multilabel_predictor.get_predictor(label_i)`

    Parameters
    ----------
    labels : List[str]
        The ith element of this list is the column (i.e. `label`) predicted by the ith TabularPredictor stored in this object.
    path : str
        Path to directory where models and intermediate outputs should be saved.
        If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
        Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
        Otherwise files from first `fit()` will be overwritten by second `fit()`.
        Caution: when predicting many labels, this directory may grow large as it needs to store many TabularPredictors.
    problem_types : List[str]
        The ith element is the `problem_type` for the ith TabularPredictor stored in this object.
    eval_metrics : List[str]
        The ith element is the `eval_metric` for the ith TabularPredictor stored in this object.
    consider_labels_correlation : bool
        Whether the predictions of multiple labels should account for label correlations or predict each label independently of the others.
        If True, the ordering of `labels` may affect resulting accuracy as each label is predicted conditional on the previous labels appearing earlier in this list (i.e. in an auto-regressive fashion).
        Set to False if during inference you may want to individually use just the ith TabularPredictor without predicting all the other labels.
    kwargs :
        Arguments passed into the initialization of each TabularPredictor.

    """

    multi_predictor_file = "multilabel_predictor.pkl"

    def __init__(
        self,
        labels,
        path,
        problem_types=None,
        eval_metrics=None,
        consider_labels_correlation=True,
        **kwargs,
    ):

        if len(labels) < 2:
            consider_labels_correlation = False
        else:
            pass
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i]: eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = self.path + "Predictor_" + label
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = self.eval_metrics[label]
            self.predictors[label] = TabularPredictor(
                label=label,
                problem_type=problem_type,
                eval_metric=eval_metric,
                path=path_i,
                **kwargs,
            )

    def fit(self, train_data, tuning_data=None, **kwargs):
        """Fits a separate TabularPredictor to predict each of the labels.

        Parameters
        ----------
        train_data, tuning_data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            See documentation for `TabularPredictor.fit()`.
        kwargs :
            Arguments passed into the `fit()` call for each TabularPredictor.
        """
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        if tuning_data is not None:
            tuning_data_og = tuning_data.copy()
        else:
            tuning_data_og = None
        save_metrics = len(self.eval_metrics) == 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                labels_to_drop = [
                    self.labels[j] for j in range(i + 1, len(self.labels))
                ]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, **kwargs):
        """Returns DataFrame with label columns containing predictions for each label.

        Parameters
        ----------
        data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            Data to make predictions for. If label columns are present in this data, they will be ignored. See documentation for `TabularPredictor.predict()`.
        kwargs :
            Arguments passed into the predict() call for each TabularPredictor.
        """
        return self._predict(data, as_proba=False, **kwargs)

    def predict_proba(self, data, **kwargs):
        """Returns dict where each key is a label and the corresponding value is the `predict_proba()` output for just that label.

        Parameters
        ----------
        data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            Data to make predictions for. See documentation for `TabularPredictor.predict()` and `TabularPredictor.predict_proba()`.
        kwargs :
            Arguments passed into the `predict_proba()` call for each TabularPredictor (also passed into a `predict()` call).
        """
        return self._predict(data, as_proba=True, **kwargs)

    def evaluate(self, data, **kwargs):
        """Returns dict where each key is a label and the corresponding value is the `evaluate()` output for just that label.

        Parameters
        ----------
        data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            Data to evalate predictions of all labels for, must contain all labels as columns. See documentation for `TabularPredictor.evaluate()`.
        kwargs :
            Arguments passed into the `evaluate()` call for each TabularPredictor (also passed into the `predict()` call).
        """
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            print(f"Evaluating TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        """Save MultilabelPredictor to disk."""
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(path=self.path + self.multi_predictor_file, object=self)
        print(
            f"MultilabelPredictor saved to disk. Load with: MultilabelPredictor.load('{self.path}')"
        )

    @classmethod
    def load(cls, path):
        """Load MultilabelPredictor from disk `path` previously specified when creating this MultilabelPredictor."""
        path = os.path.expanduser(path)
        if path[-1] != os.path.sep:
            path = path + os.path.sep
        return load_pkl.load(path=path + cls.multi_predictor_file)

    def redirect_predictor(self, old_path):
        base_path = "Predictor_".join(old_path.split("Predictor_")[:-1])
        base_old_dirs = []
        for sf in [POOL_SUBFOLDER, ESTIMATORS_SUBFOLDER]:
            if sf in base_path:
                strip = sf.join(base_path.split(sf)[:-1])
                base_old_dirs += [strip]
        base_old_dir = None
        l = 0
        for bod in base_old_dirs:
            l = len(bod)
            if base_old_dir is None:
                base_old_dir = bod
            else:
                if l > len(base_old_dir):
                    base_old_dir = bod
                else:
                    continue
        new_path = old_path.replace(base_old_dir, get_model_dir() + "/")
        return new_path

    def get_predictor(self, label):
        """Returns TabularPredictor which is used to predict this label."""
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            predictor = self.redirect_predictor(predictor)
            return TabularPredictor.load(path=predictor)
        return predictor

    def _get_data(self, data):
        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()

    def _predict(self, data, as_proba=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            print(f"Predicting with TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(
                    data, as_multiclass=True, **kwargs
                )
            data[label] = predictor.predict(data, **kwargs)
        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict
