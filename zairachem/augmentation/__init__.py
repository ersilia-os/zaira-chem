from .. import ZairaBase
from ..descriptors.baseline import Embedder


class MoleculeSampler(ZairaBase):
    def __init__(self, path, generative):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path
        self.generative = generative

    def sample_from_locally_available_molecules(self):
        pass

    def sample_with_generative_model(self):
        pass

    def sample(self, n):
        if self.generative:
            self.sample_with_generative_model()
        else:
            self.sample_from_locally_available_molecules()


class TabularDataset(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def _get_y(self):
        pass

    def _get_X(self):
        pass

    def get(self):
        pass


class SemiSupervised(ZairaBase):
    def __init__(self, path):
        ZairaBase.__init__(self)
        if path is None:
            self.path = self.get_output_dir()
        else:
            self.path = path

    def fit(self):
        pass

    def impute(self):
        pass

    def write(self):
        pass

    def run(self):
        self.fit()
        self.impute()
        self.save()
