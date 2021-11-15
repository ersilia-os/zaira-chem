import click

from . import zairachem_cli
from ..echo import echo

from ...fit.fit import Fitter


def fit_cmd():
    @zairachem_cli.command(help="Fit the data")
    @click.option("--dir", "-d", type=click.STRING)
    def fit(dir):
        echo("Fitting from {0}".format(dir))
        ft = Fitter(path=dir)
        ft.run()
        echo("Done", fg="green")
