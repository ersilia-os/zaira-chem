import os


class PredictSetup(object):
    def __init__(self):
        pass

    def _make_output_dir(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    def setup(self):
        pass
