from .commands.fit import fit_cmd
from .commands.predict import predict_cmd


class Command(object):

    def __init__(self):
        pass

    def fit(self):
        fit_cmd()

    def predict(self):
        predict_cmd()
