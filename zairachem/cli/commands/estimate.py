import click

from . import zairachem_cli
from ..echo import echo

from ...estimators.estimate import Estimator
from ...estimators.assemble import OutcomeAssembler
from ...estimators.performance import PerformanceReporter


def estimate_cmd():
    @zairachem_cli.command(help="Run estimators")
    @click.option("--dir", "-d", type=click.STRING)
    def estimate(dir):
        echo("Estimator".format(dir))
        e = Estimator(path=dir)
        e.run()
        o = OutcomeAssembler(path=dir)
        o.run()
        p = PerformanceReporter(path=dir)
        p.run()
        echo("Done", fg="green")
