import os

from .raw import RawDescriptors
from .treated import TreatedDescriptors
from .reference import ReferenceDescriptors, SimpleDescriptors
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
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)
        self.logger.debug(self.path)

    def _raw_descriptions(self):
        if self.is_lazy:
            self.logger.info("Lazy mode skips raw descriptors")
            return
        step = PipelineStep("raw_descriptions", self.output_dir)
        if not step.is_done():
            RawDescriptors().run()
            step.update()

    def _treated_descriptions(self):
        if self.is_lazy:
            return
        step = PipelineStep("treated_descriptions", self.output_dir)
        if not step.is_done():
            TreatedDescriptors().run()
            step.update()

    def _reference_descriptors(self):
        step = PipelineStep("reference_descriptors", self.output_dir)
        if not step.is_done():
            if self.is_lazy:
                SimpleDescriptors().run()
            else:
                ReferenceDescriptors().run()
            step.update()

    def _manifolds(self):
        step = PipelineStep("manifolds", self.output_dir)
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
