import click
from zairachem import create_session_symlink
import sys

from . import zairachem_cli
from ..echo import echo

from ...utils.pipeline import SessionFile
from ... import create_session_symlink


def session_cmd():
    @zairachem_cli.command(help="Set up a new session")
    @click.option("--output_dir", "-o", default=None, type=click.STRING)
    @click.option("--model_dir", "-m", default=None, type=click.STRING)
    @click.option("--fit/--predict", default=True)
    def session(output_dir, model_dir, fit):
        echo("Opening a session")
        if fit:
            mode = "train"
            if model_dir is None:
                sys.exit(1)
            output_dir = model_dir
        else:
            mode = "predict"
            if output_dir is None:
                sys.exit(1)
        sf = SessionFile(output_dir)
        sf.open_session(mode, output_dir, model_dir)
        create_session_symlink(output_dir)
