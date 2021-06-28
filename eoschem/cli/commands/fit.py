import click

from . import eoschem_cli
from .. import echo

from ...train.train import Trainer


def fit_cmd():
    @eoschem_cli.command(help="Fit the data")
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option("--output_dir", "-o", default="output", type=click.STRING)
    @click.option("--minutes", default=60, type=click.INT)
    def fit(k):
        echo("Fitting from {0}".format(input_file))
        tr = Trainer(input_file=input_file, output_dir=output_dir)
        tr.train()
        echo("Results {0}".format(output_folder))
        echo("Done", fg="green")
