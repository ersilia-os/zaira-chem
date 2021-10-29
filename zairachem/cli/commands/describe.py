import click

from . import zairachem_cli
from ..echo import echo
from ...descriptors.describe import Describer


def describe_cmd():
    @zairachem_cli.command(help="Calculate descriptors and normalize them")
    @click.option("--dir", "-d", default=None, type=click.STRING)
    def describe(dir):
        echo("Calculating descriptors".format(dir))
        desc = Describer(path=dir)
        desc.run()
        echo("Done", fg="green")
