from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


TEST_SIZE = 0.66

class Confidence(object):

    def __init__(self):
        pass

    def _is_classification(self, y):
        if len(set(y)) == 2:
            return True
        else:
            return False

    def fit(self, X, y):
        is_clf = self._is_classification(y)
        if is_clf:
            X_train, X_conf, y_train, y_conf = train_test_split(X, y, stratify=y, test_size=TEST_SIZE)
            self.mdl =
            self.mac = MacestClassification()
        else:
            macest = MacestRegression()


    def predict(self):
        pass
