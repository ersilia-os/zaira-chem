import click

from . import eoschem_cli
from ..echo import echo

from ...setup.setup import Setup


def setup_cmd():
    @eoschem_cli.command(help="Setup ML task")
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option("--output_dir", "-o", default="output", type=click.STRING)
    @click.option("--time_budget", "-t", default=60, type=click.INT)
    def setup(input_file, output_dir, time_budget):
        echo("Reading from {0}".format(input_file))
        s = Setup(input_file=input_file, output_dir=output_dir, time_budget=time_budget)
        s.setup()
        echo("Results will be stored at {0}".format(output_dir))
        echo("Done", fg="green")
