from .prepare import Prepare
from .standardize import Standardize
from .descriptors import Descriptors
from .folding import Folding
from .aggregate import AggregateActivity
from .thresholding import Thresholding


class MelloddyTunerTrainPipeline(object):
    def __init__(self, path):
        self.path = path

    def run(self):
        Prepare(self.path).run()
        Standardize(self.path).run()
        Descriptors(self.path).run()
        Folding(self.path).run()


#        AggregateActivity(self.path).run()
#        Thresholding(self.path).run()

class MelloddyTunerPredictPipelien(object):
    def __init__(self, path):
        self.path = path

    def run(self):
        Prepare(self.path).run()
        Standardize(self.path).run()
