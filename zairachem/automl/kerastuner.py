import os
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from autokeras.tuners.hyperband import keras_tuner as kt
import autokeras as ak
from tensorflow.keras.models import load_model


TUNER_PROJECT_NAME = "kerastuner"
COLUMNS_FILENAME = "columns.json"

EPOCHS = 100
VALIDATION_SPLIT = 0.2


class BaseTunerTrainer(object):
    def __init__(self, X, y, save_path):
        self.X = X
        self.y = y
        self.input_shape = X.shape[1]
        self.output_shape = y.shape[1]
        self.save_path = save_path

    def _final_train(self, X, y):
        self.hypermodel = self.tuner.hypermodel.build(self.best_hps)
        self.hypermodel.fit(
            X, y, epochs=self.best_epoch, validation_split=VALIDATION_SPLIT
        )

    def fit(self):
        self._search(self.X, self.y)
        self._get_best_epoch(self.X, self.y)
        self._final_train(self.X, self.y)

    def save(self):
        print("Saving")
        self.hypermodel.save(
            os.path.join(self.save_path, TUNER_PROJECT_NAME, self.task)
        )

    def export_model(self):
        return self.hypermodel


class TunerTrainerRegressor(BaseTunerTrainer):
    def __init__(self, X, y, save_path):
        BaseTunerTrainer.__init__(self, X, y, save_path)
        print("Is regression")
        self.loss = "mean_squared_error"
        self.task = "reg"
        self.metrics = [keras.metrics.RootMeanSquaredError()]
        self.objective = "val_loss"

    def _model_builder(self, hp):
        model = keras.Sequential()
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.add(
            keras.layers.Dense(
                units=hp_units, activation="relu", input_shape=(self.input_shape,)
            )
        )
        model.add(keras.layers.Dense(self.output_shape))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return model

    def _search(self, X, y):
        self.tuner = kt.Hyperband(
            self._model_builder,
            objective=self.objective,
            max_epochs=10,
            factor=3,
            directory=os.path.join(self.save_path, TUNER_PROJECT_NAME, self.task),
            project_name="trials",
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        self.tuner.search(
            X,
            y,
            epochs=100,
            validation_split=VALIDATION_SPLIT,
            callbacks=[stop_early],
            verbose=True,
        )
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

    def _get_best_epoch(self, X, y):
        model = self.tuner.hypermodel.build(self.best_hps)
        history = model.fit(X, y, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
        val_per_epoch = history.history[self.objective]
        self.best_epoch = val_per_epoch.index(min(val_per_epoch)) + 1
        print("Best epoch: %d" % (self.best_epoch,))


class TunerTrainerClassifier(BaseTunerTrainer):
    def __init__(self, X, y, save_path):
        BaseTunerTrainer.__init__(self, X, y, save_path)
        print("Is classification")
        self.loss = "binary_crossentropy"
        self.task = "clf"
        self.metrics = [keras.metrics.AUC()]
        self.objective = "val_auc"

    def _model_builder(self, hp):
        model = keras.Sequential()
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.add(
            keras.layers.Dense(
                units=hp_units, activation="relu", input_shape=(self.input_shape,)
            )
        )
        model.add(keras.layers.Dense(self.output_shape, activation="sigmoid"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=self.loss,
            metrics=self.metrics,
        )
        return model

    def _search(self, X, y):
        self.tuner = kt.Hyperband(
            self._model_builder,
            objective=kt.Objective(self.objective, direction="max"),
            max_epochs=10,
            factor=3,
            directory=os.path.join(self.save_path, TUNER_PROJECT_NAME, self.task),
            project_name="trials",
        )
        stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
        self.tuner.search(
            X,
            y,
            epochs=100,
            validation_split=VALIDATION_SPLIT,
            callbacks=[stop_early],
            verbose=True,
        )
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

    def _get_best_epoch(self, X, y):
        model = self.tuner.hypermodel.build(self.best_hps)
        history = model.fit(X, y, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)
        val_per_epoch = history.history[self.objective]
        self.best_epoch = val_per_epoch.index(max(val_per_epoch)) + 1
        print("Best epoch: %d" % (self.best_epoch,))


class KerasTunerEstimator(object):
    def __init__(self, save_path):
        self.save_path = save_path
        self.reg_estimator = None
        self.clf_estimator = None

    def _coltype_splitter(self, data, labels):
        x_cols = []
        clf_cols = []
        reg_cols = []
        for c in list(data.columns):
            if c not in labels:
                x_cols += [c]
            else:
                if "clf" in c:
                    clf_cols += [c]
                else:
                    reg_cols += [c]
        self.columns = {"X": x_cols, "reg": reg_cols, "clf": clf_cols, "labels": labels}
        data_x = data[x_cols]
        data_clf = data[clf_cols]
        data_reg = data[reg_cols]
        return data_x, data_clf, data_reg

    def fit(self, data, labels):
        data_x, data_clf, data_reg = self._coltype_splitter(data, labels)
        clf_cols = list(data_clf)
        reg_cols = list(data_reg)
        X = np.array(data_x)
        if reg_cols:
            self.reg_estimator = TunerTrainerRegressor(
                X, np.array(data_reg), save_path=self.save_path
            )
            self.reg_estimator.fit()
        else:
            self.reg_estimator = None
        if clf_cols:
            self.clf_estimator = TunerTrainerClassifier(
                X, np.array(data_clf), save_path=self.save_path
            )
            self.clf_estimator.fit()
        else:
            self.clf_estimator = None

    def save(self):
        if self.reg_estimator is not None:
            print("Saving reg estimator")
            self.reg_estimator.save()
        if self.clf_estimator is not None:
            print("Saving clf estimator")
            self.clf_estimator.save()
        with open(os.path.join(self.save_path, COLUMNS_FILENAME), "w") as f:
            json.dump(self.columns, f)

    def load(self):
        with open(os.path.join(self.save_path, COLUMNS_FILENAME), "r") as f:
            columns = json.load(f)
        reg_path = os.path.join(self.save_path, TUNER_PROJECT_NAME, "reg")
        if os.path.exists(reg_path):
            reg_estimator = load_model(reg_path, custom_objects=ak.CUSTOM_OBJECTS)
        else:
            reg_estimator = None
        clf_path = os.path.join(self.save_path, TUNER_PROJECT_NAME, "clf")
        if os.path.exists(clf_path):
            clf_estimator = load_model(clf_path, custom_objects=ak.CUSTOM_OBJECTS)
        else:
            clf_estimator = None
        return KerasTunerArtifact(
            reg_estimator=reg_estimator, clf_estimator=clf_estimator, columns=columns
        )


class KerasTunerArtifact(object):
    def __init__(self, reg_estimator, clf_estimator, columns):
        self.reg_estimator = reg_estimator
        self.clf_estimator = clf_estimator
        self.columns = columns

    def predict(self, data):
        X = np.array(data[self.columns["X"]])
        if self.reg_estimator is not None:
            print(self.reg_estimator.summary())
            y_reg = self.reg_estimator.predict(X)
        if self.clf_estimator is not None:
            print(self.clf_estimator.summary())
            y_clf = self.clf_estimator.predict_proba(X)
            y_clf_bin = np.zeros(y_clf.shape)
            y_clf_bin[y_clf > 0.5] = 1
        P = []
        labels = []
        for label in self.columns["labels"]:
            if "clf" in label and self.clf_estimator is not None:
                idx = self.columns["clf"].index(label)
                P += [list(y_clf[:, idx])]
                P += [list(y_clf_bin[:, idx])]
                labels += [label, label + "_bin"]
                continue
            if "reg" in label and self.reg_estimator is not None:
                idx = self.columns["reg"].index(label)
                P += [list(y_reg[:, idx])]
                labels += [label]
        P = np.array(P).T
        df = pd.DataFrame(P, columns=labels)
        return df

    def run(self, data):
        results = self.predict(data)
        return results
