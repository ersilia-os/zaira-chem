from .raw import RawDescriptors
from .treated import TreatedDescriptors
from .reference import ReferenceDescriptors
from .manifolds import Manifolds

from .. import ZairaBase

from ..utils.pipeline import PipelineStep


class Describer(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.logger.debug(self.path)

    def _raw_descriptions(self):
        step = PipelineStep("raw_descriptions")
        if not step.is_done():
            RawDescriptors().run()
            step.update()

    def _treated_descriptions(self):
        step = PipelineStep("treated_descriptions")
        if not step.is_done():
            TreatedDescriptors().run()
            step.update()

    def _reference_descriptors(self):
        step = PipelineStep("reference_descriptors")
        if not step.is_done():
            ReferenceDescriptors().run()
            step.update()

    def _manifolds(self):
        step = PipelineStep("manifolds")
        if not step.is_done():
            Manifolds().run()
            step.update()

    def run(self):
        self.reset_time()
        self._raw_descriptions()
        self._treated_descriptions()
        self._reference_descriptors()
        self._manifolds()
        self.update_elapsed_time()
