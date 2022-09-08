from .estimate import Estimator
from .assemble import OutcomeAssembler
from .performance import PerformanceReporter


class IndividualReducedDescriptorPipeline(object):
    def __init__(self, path):
        self.e = Estimator(path=path)
        self.a = OutcomeAssembler(path=path)
        self.p = PerformanceReporter(path=path)

    def run(self, time_budget_sec=None):
        self.e.run(time_budget_sec=time_budget_sec)
        self.a.run()
        self.p.run()
