import click

from . import zairachem_cli
from ..echo import echo

from ...pool.pool import Pool


def pool_cmd():
    @zairachem_cli.command(help="Pool ensemble of classifiers")
    @click.option("--dir", "-d", type=click.STRING)
    def pool(dir):
        echo("Pooling from {0}".format(dir))
        pool = Pool(dir)
        pool.pool()
        echo("Done", fg="green")
