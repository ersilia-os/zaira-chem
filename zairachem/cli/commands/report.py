import click

from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo
from ...reports.report import Reporter


def report_cmd():
    @zairachem_cli.command(help="Reports of the model performance and results")
    @click.option("--dir", "-d", type=click.STRING)
    @click.option(
        "--plot_name",
        "-p",
        type=click.STRING,
        default=None,
        help="Only do the specified plot",
    )
    def report(dir, plot_name):
        create_session_symlink(dir)
        echo("Report".format(dir))
        r = Reporter(path=dir, plot_name=plot_name)
        r.run()
        echo("Done", fg="green")
