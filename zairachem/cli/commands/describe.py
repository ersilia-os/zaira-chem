import click

from . import zairachem_cli
from ..echo import echo

from ...descriptors.descriptors import Descriptors


def describe_cmd():
    @zairachem_cli.command(help="Calculate descriptors")
    @click.option("--dir", "-d", type=click.STRING)
    def describe(dir):
        echo("Calculating descriptors in {0}".format(dir))
        desc = Descriptors(dir=dir)
        desc.calculate()
        echo("Done", fg="green")
