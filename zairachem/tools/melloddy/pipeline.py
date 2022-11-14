from .prepare import PrepareTrain, PreparePredict
from .standardize import Standardize
from .descriptors import Descriptors
from .folding import Folding
from .aggregate import AggregateActivity
from .thresholding import Thresholding


class MelloddyTunerTrainPipeline(object):
    def __init__(self, path):
        self.path = path

    def run(self):
        PrepareTrain(self.path).run()
        Standardize(self.path).run()
        Descriptors(self.path).run()
        Folding(self.path).run()


#        AggregateActivity(self.path).run() # TODO
#        Thresholding(self.path).run() # TODO


class MelloddyTunerPredictPipeline(object):
    def __init__(self, path):
        self.path = path

    def run(self, has_tasks):
        if has_tasks:
            PrepareTrain(self.path).run()
        PreparePredict(self.path).run()
        Standardize(self.path).run()
