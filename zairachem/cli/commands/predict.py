import click

from . import zairachem_cli
from ..echo import echo

from ...setup.prediction import PredictSetup
from ...descriptors.describe import Describer
from ...estimators.estimate import Estimator
from ...estimators.assemble import OutcomeAssembler
from ...estimators.performance import PerformanceReporter
from ...plots.plot import Plotter


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
        e = Estimator(path=output_dir)
        e.run()
        o = OutcomeAssembler(path=output_dir)
        o.run()
        p = PerformanceReporter(path=output_dir)
        p.run()
        p = Plotter(path=output_dir)
        p.run()
        echo("Done", fg="green")
