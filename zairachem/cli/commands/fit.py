import click

from . import zairachem_cli
from ..echo import echo

from ...estimators.estimate import Estimator


def estimate_cmd():
    @zairachem_cli.command(help="Run estimators")
    @click.option("--dir", "-d", type=click.STRING)
    def estimate(dir):
        echo("Estimator".format(dir))
        ft = Estimator(path=dir)
        ft.run()
        echo("Done", fg="green")
