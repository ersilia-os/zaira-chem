from ... import logger
import numpy as np
import time


class BaseDescriptorType(object):

    def __init__(self):
        self.name = self.__class__.__name__.lower()
        super().__init__()

    def calc(self, mols):
        logger.debug("Calculating descriptor `%s` " % self.name +
                     "for %i molecules." % len(mols))
        t0 = time.time()
        calculated = self._calc(mols)
        t1 = time.time()
        if not isinstance(calculated, np.ndarray):
            raise Exception("Descriptor returned with wrong type:" +
                            str(type(calculated)))
        if not calculated.ndim == 2:
            raise Exception("Descriptor returned with wrong dimension:" +
                            str(calculated.ndim))
        logger.debug("Calculated descriptor `%s` " % self.name +
                     "with shape %s. " % str(calculated.shape) +
                     "Took: %i secs." % int(t1 - t0))
        return calculated
