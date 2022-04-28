import click

from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo
from ...finish.finisher import Finisher


def finish_cmd():
    @zairachem_cli.command(help="Finish pipeline")
    @click.option("--dir", "-d", type=click.STRING)
    def finish(dir):
        create_session_symlink(dir)
        echo("Finishing".format(dir))
        r = Finisher(path=dir)
        r.run()
        echo("Done", fg="green")
