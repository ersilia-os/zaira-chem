import click

from . import zairachem_cli
from ..echo import echo

from ...setup.training import TrainSetup
from ...descriptors.describe import Describer
from ...estimators.estimate import Estimator
from ...estimators.assemble import OutcomeAssembler
from ...estimators.performance import PerformanceReporter


def fit_cmd():
    @zairachem_cli.command(help="Fit an ML-based QSAR model")
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option("--output_dir", "-o", default="output", type=click.STRING)
    @click.option("--parameters", "-p", default=None, type=click.STRING)
    def fit(input_file, output_dir, parameters):
        echo("Results will be stored at {0}".format(output_dir))
        s = TrainSetup(
            input_file=input_file,
            output_dir=output_dir,
            parameters=parameters,
            time_budget=60 # TODO
        )
        s.setup()
        d = Describer(
            path=output_dir
        )
        d.run()
        e = Estimator(
            path=output_dir
        )
        e.run()
        o = OutcomeAssembler(
            path=output_dir
        )
        o.run()
        p = PerformanceReporter(
            path=output_dir
        )
        p.run()
        echo("Done", fg="green")
