import click
import onnx

from . import zairachem_cli
from ..echo import echo

from ...setup.prediction import ZairaPredictSetup, ONNXPredictSetup
from ...descriptors.describe import Describer
from ...estimators.pipe import EstimatorPipeline
from ...pool.pool import Pooler
from ...applicability.applicability import ApplicabilityEvaluator
from ...reports.report import Reporter
from ...finish.finisher import Finisher
from ...tools.onnx.predict import ONNXPredictor


def predict_cmd():
    @zairachem_cli.command(help="Make predictions")
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option("--output_dir", "-o", default=None, type=click.STRING)
    @click.option("--model_dir", "-m", default=None, type=click.STRING)
    @click.option(
        "--clean",
        is_flag=True,
        show_default=True,
        default=False,
        help="Clean directory at the end of the pipeline. Only precalculated descriptors are removed.",
    )
    @click.option(
        "--flush",
        is_flag=True,
        show_default=True,
        default=False,
        help="Flush directory at the end of the pipeline. Only data, results and reports are kept. Use with caution: the original trained model will be flushed too.",
    )
    def predict(input_file, output_dir, model_dir, clean, flush):
        echo("Results will be stored at {0}".format(output_dir))
        # check if model is Olinda onnx model 
        try:
            onnx.load(model_dir)
            is_onnx = True
        except:
            is_onnx = False
            
        if is_onnx:
            s = ONNXPredictSetup(
                input_file=input_file,
                output_dir=output_dir,
                model_dir=model_dir,
                time_budget=60,  # TODO
            )
            if s.is_done():
                echo("Results are already available. Skipping calculations")
                return
            s.setup()
            
            op = ONNXPredictor(model_dir, output_dir)
            op.run()
            
        else:
            s = ZairaPredictSetup(
                input_file=input_file,
                output_dir=output_dir,
                model_dir=model_dir,
                time_budget=60,  # TODO
            )    
            if s.is_done():
                echo("Results are already available. Skipping calculations")
                return
            s.setup()
        
            d = Describer(path=output_dir)
            d.run()
            e = EstimatorPipeline(path=output_dir)
            e.run()
            p = Pooler(path=output_dir)
            p.run()
            a = ApplicabilityEvaluator(path=output_dir)
            a.run()
            r = Reporter(path=output_dir)
            r.run()
            f = Finisher(path=output_dir, clean=clean, flush=flush)
            f.run()
            echo("Done", fg="green")
