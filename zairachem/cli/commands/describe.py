import click
from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo
from ...descriptors.describe import Describer


def describe_cmd():
    @zairachem_cli.command(help="Calculate descriptors and normalize them")
    @click.option("--dir", "-d", type=click.STRING)
    def describe(dir):
        create_session_symlink(dir)
        echo("Calculating descriptors")
        desc = Describer(path=dir)
        desc.run()
        echo("Done", fg="green")
