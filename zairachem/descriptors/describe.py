from .raw import RawDescriptors
from .treated import TreatedDescriptors
from .reference import ReferenceDescriptors
from .manifolds import Manifolds

from .. import ZairaBase


class Describer(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.logger.debug(self.path)

    def _raw_descriptions(self):
        RawDescriptors().run()

    def _treated_descriptions(self):
        TreatedDescriptors().run()

    def _reference_descriptors(self):
        ReferenceDescriptors().run()

    def _manifolds(self):
        Manifolds().run()

    def run(self):
        self.reset_time()
        self._raw_descriptions()
        self._treated_descriptions()
        self._reference_descriptors()
        self._manifolds()
        self.update_elapsed_time()
