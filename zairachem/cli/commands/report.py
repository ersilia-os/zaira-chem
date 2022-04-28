import click

from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo
from ...reports.report import Reporter


def report_cmd():
    @zairachem_cli.command(help="Reports of the model performance and results")
    @click.option("--dir", "-d", type=click.STRING)
    def report(dir):
        create_session_symlink(dir)
        echo("Report".format(dir))
        r = Reporter(path=dir)
        r.run()
        echo("Done", fg="green")
