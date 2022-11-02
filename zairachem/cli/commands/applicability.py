import click
from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo
from ...applicability.applicability import ApplicabilityEvaluator


def applicability_cmd():
    @zairachem_cli.command(help="Evaluate domain of applicability of the model")
    @click.option("--dir", "-d", type=click.STRING)
    def applicability(dir):
        create_session_symlink(dir)
        echo("Evaluating applicability")
        appl = ApplicabilityEvaluator(path=dir)
        appl.run()
        echo("Done", fg="green")
