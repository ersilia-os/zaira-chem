import click
from zairachem import create_session_symlink

from . import zairachem_cli
from ..echo import echo

from ...utils.pipeline import SessionFile
from ... import create_session_symlink


def session_cmd():
    @zairachem_cli.command(help="Set up a new session")
    @click.option("--output_dir", "-o", type=click.STRING)
    @click.option("--model_dir", "-m", default=None, type=click.STRING)
    @click.option("--train/--predict", default=True)
    def session(output_dir, model_dir, train):
        echo("Opening a session")
        if train:
            mode = "train"
            model_dir = output_dir
        else:
            mode = "predict"
        sf = SessionFile(output_dir)
        sf.open_session(mode, output_dir, model_dir)
        create_session_symlink(output_dir)
