from .commands.session import session_cmd
from .commands.setup import setup_cmd
from .commands.describe import describe_cmd
from .commands.estimate import estimate_cmd
from .commands.report import report_cmd
from .commands.pool import pool_cmd
from .commands.finish import finish_cmd
from .commands.fit import fit_cmd
from .commands.predict import predict_cmd
from .commands.example import example_cmd
from .commands.applicability import applicability_cmd


class Command(object):
    def __init__(self):
        pass

    def session(self):
        session_cmd()

    def setup(self):
        setup_cmd()

    def describe(self):
        describe_cmd()

    def estimate(self):
        estimate_cmd()

    def pool(self):
        pool_cmd()

    def applicability(self):
        applicability_cmd()

    def report(self):
        report_cmd()

    def finish(self):
        finish_cmd()

    def fit(self):
        fit_cmd()

    def predict(self):
        predict_cmd()

    def example(self):
        example_cmd()
