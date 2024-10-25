import os

from olinda import distillation
from .. import ZairaBase

class Distiller(ZairaBase):
    def __init__(self, zaira_path, onnx_output_path):
        ZairaBase.__init__(self)
        if zaira_path is None:
            self.zaira_path = self.get_output_dir()
        else:
            self.zaira_path = zaira_path
        self.trained_dir = os.path.abspath(self.zaira_path)
        assert os.path.exists(self.trained_dir)
        
        self.onnx_output_path = onnx_output_path
        
    def run(self):
        olinda_distiller = distillation.Distiller()
        onnx_model = olinda_distiller.distill(self.trained_dir)
        onnx_model.save(self.onnx_output_path)
