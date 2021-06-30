import click

from . import eoschem_cli
from ..echo import echo

from ...fit.fit import Fit


def fit_cmd():
    @eoschem_cli.command(help="Fit the data")
    @click.option("--dir", "-d", type=click.STRING)
    def fit(dir):
        echo("Fitting from {0}".format(dir))
        ft = Fit(dir)
        ft.fit()
        echo("Done", fg="green")
