import os
import pandas as pd
import shutil

from ... import ZairaBase
from ...vars import DATA_SUBFOLDER, DATA_FILENAME, OUTPUT_FILENAME, RESULTS_FILENAME, REPORT_SUBFOLDER

import onnx_runner

class ONNXPredictor(ZairaBase):
    def __init__(self, model_path, output_path):
        ZairaBase.__init__(self)
        self.output_dir = output_path
        assert os.path.exists(self.output_dir)
        self.model_path = model_path        
        self.data = pd.read_csv(
            os.path.join(self.output_dir, DATA_SUBFOLDER, DATA_FILENAME)
        )
    
    def run(self):
        model = onnx_runner.onnx_runner(self.model_path)
        input_path = os.path.join(self.output_dir, DATA_SUBFOLDER, DATA_FILENAME)
        input_df = pd.read_csv(input_path)
        
        input_smiles = input_df["smiles"].tolist()
        preds = model.predict(input_smiles)
        preds_bin = [1 if val > 0.5 else 0.0 for val in preds]
        
        output_df = pd.DataFrame(list(zip(input_smiles, preds, preds_bin)), 
            columns = ["smiles", "clf_distill", "clf_distill_bin"])
        output_path = os.path.join(self.output_dir, REPORT_SUBFOLDER, RESULTS_FILENAME)
        output_df.to_csv(output_path, index=False)
        
        shutil.copy(output_path, os.path.join(self.output_dir, "output.csv"))
        
        
