import click

from . import zairachem_cli
from ..echo import echo

import pandas as pd

from ...datasets import make_classification, make_regression


def example_cmd():
    @zairachem_cli.command(help="Create an example")
    @click.option("--file_name", "-f", type=click.STRING)
    @click.option("--n_samples", "-n", default=1000, type=click.INT)
    @click.option("--proportion", "-p", default=0.1, type=click.FLOAT)
    @click.option("--regression/--classification", default=True)
    def example(file_name, regression, n_samples, proportion):
        echo("Creating an example")
        if regression:
            data = make_regression(n_samples=n_samples)
        else:
            data = make_classification(n_samples=n_samples, p=proportion)
        data = {"smiles": data[0], "activity": data[1]}
        data = pd.DataFrame(data)
        data.to_csv(file_name, index=False)
