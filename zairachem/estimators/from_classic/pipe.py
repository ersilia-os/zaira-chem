from .estimate import Estimator


class ClassicPipeline(object):
    def __init__(self, path):
        self.e = Estimator(path=path)

    def run(self, time_budget_sec=None):
        self.e.run(time_budget_sec=time_budget_sec)
