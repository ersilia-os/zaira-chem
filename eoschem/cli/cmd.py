from .commands.setup import setup_cmd
from .commands.describe import describe_cmd
from .commands.fit import fit_cmd
from .commands.pool import pool_cmd
from .commands.predict import predict_cmd


class Command(object):
    def __init__(self):
        pass

    def setup(self):
        setup_cmd()

    def describe(self):
        describe_cmd()

    def fit(self):
        fit_cmd()

    def pool(self):
        pool_cmd()

    def predict(self):
        predict_cmd()
