import click

from . import zairachem_cli
from ..echo import echo

from ...setup.prediction import PredictSetup
from ...descriptors.describe import Describer
from ...estimators.pipe import EstimatorPipeline
from ...pool.pool import Pooler
from ...reports.report import Reporter
from ...finish.finisher import Finisher


def predict_cmd():
    @zairachem_cli.command(help="Make predictions")
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option("--output_dir", "-o", default=None, type=click.STRING)
    @click.option("--model_dir", "-m", default=None, type=click.STRING)
    def predict(input_file, output_dir, model_dir):
        echo("Results will be stored at {0}".format(output_dir))
        s = PredictSetup(
            input_file=input_file,
            output_dir=output_dir,
            model_dir=model_dir,
            time_budget=60,  # TODO
        )
        s.setup()
        d = Describer(path=output_dir)
        d.run()
        e = EstimatorPipeline(path=output_dir)
        e.run()
        p = Pooler(path=output_dir)
        p.run()
        r = Reporter(path=output_dir)
        r.run()
        f = Finisher(path=output_dir)
        f.run()
        echo("Done", fg="green")
