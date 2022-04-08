import click

from . import zairachem_cli
from ..echo import echo

from ... import open_session


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
        open_session(output_dir, model_dir, mode)
