import click

from . import zairachem_cli
from ..echo import echo
from ...plots.plot import Plotter


def plot_cmd():
    @zairachem_cli.command(help="Diagnostics plots")
    @click.option("--dir", "-d", type=click.STRING)
    def plot(dir):
        echo("Plot".format(dir))
        p = Plotter(path=dir)
        p.run()
        echo("Done", fg="green")
