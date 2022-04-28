import click

from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo

from ...pool.pool import Pooler


def pool_cmd():
    @zairachem_cli.command(help="Pool ensemble of estimators")
    @click.option("--dir", "-d", type=click.STRING)
    def pool(dir):
        create_session_symlink(dir)
        echo("Pooling from {0}".format(dir))
        pl = Pooler(dir)
        pl.run()
        echo("Done", fg="green")
