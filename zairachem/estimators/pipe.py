import os

from .. import ZairaBase
from ..utils.pipeline import PipelineStep
from .from_classic.pipe import ClassicPipeline
from .from_fingerprint.pipe import FingerprintPipeline
from .from_individual_full_descriptors.pipe import IndividualFullDescriptorPipeline
from .from_manifolds.pipe import ManifoldPipeline
from .from_reference_embedding.pipe import ReferenceEmbeddingPipeline
from .from_molmap.pipe import MolMapPipeline
from .evaluate import SimpleEvaluator


class EstimatorPipeline(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.output_dir = os.path.abspath(self.path)
        assert os.path.exists(self.output_dir)

    def _classic_estimator_pipeline(self, time_budget_sec):
        step = PipelineStep("classic_estimator_pipeline", self.output_dir)
        if not step.is_done():
            self.logger.debug("Running classic estimator pipeline")
            p = ClassicPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _fingerprint_estimator_pipeline(self, time_budget_sec):
        step = PipelineStep("fingerprint_estimator_pipeline", self.output_dir)
        if not step.is_done():
            self.logger.debug("Running fingerprint estimator pipeline")
            p = FingerprintPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _individual_estimator_pipeline(self, time_budget_sec):
        step = PipelineStep("individual_estimator_pipeline", self.output_dir)
        if not step.is_done():
            self.logger.debug("Running individual estimator pipeline")
            p = IndividualFullDescriptorPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _manifolds_pipeline(self, time_budget_sec):
        step = PipelineStep("manifolds_pipeline", self.output_dir)
        if not step.is_done():
            self.logger.debug("Running manifolds estimator pipeline")
            p = ManifoldPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _reference_pipeline(self, time_budget_sec):
        step = PipelineStep("reference_pipeline", self.output_dir)
        if not step.is_done():
            self.logger.debug("Reference embedding pipeline")
            p = ReferenceEmbeddingPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _molmap_pipeline(self, time_budget_sec):
        step = PipelineStep("molmap_pipeline", self.output_dir)
        if not step.is_done():
            self.logger.debug("Molmap estimator pipeline")
            p = MolMapPipeline(path=self.path)
            p.run(time_budget_sec=time_budget_sec)
            step.update()

    def _simple_evaluation(self):
        step = PipelineStep("simple_evaluation", self.output_dir)
        if not step.is_done():
            SimpleEvaluator(path=self.path).run()
            step.update()

    def run(self, time_budget_sec=None):
        self._classic_estimator_pipeline(time_budget_sec)
        self._fingerprint_estimator_pipeline(time_budget_sec)
        self._individual_estimator_pipeline(time_budget_sec)
        self._manifolds_pipeline(time_budget_sec)
        self._reference_pipeline(time_budget_sec)
        self._molmap_pipeline(time_budget_sec)
        self._simple_evaluation()
