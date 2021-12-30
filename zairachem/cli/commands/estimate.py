import click

from . import zairachem_cli
from ..echo import echo
from ...estimators.pipe import EstimatorPipeline


def estimate_cmd():
    @zairachem_cli.command(help="Run estimators")
    @click.option("--dir", "-d", type=click.STRING)
    @click.option(
        "--time-budget",
        "-t",
        default=None,
        type=click.INT,
        help="Time budget in seconds for each of the two models.",
    )
    def estimate(dir, time_budget):
        echo("Estimator".format(dir))
        e = EstimatorPipeline(path=dir)
        e.run(time_budget_sec=time_budget)
        echo("Done", fg="green")
