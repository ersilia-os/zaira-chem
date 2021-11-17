import click

from . import zairachem_cli
from ..echo import echo

from ...setup.training import TrainSetup
from ...setup.prediction import PredictSetup


def setup_cmd():
    @zairachem_cli.command(help="Setup Zaira machine learning task")
    @click.argument("task", type=click.STRING)
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option("--output_dir", "-o", default="output", type=click.STRING)
    @click.option("--model_dir", "-m", default=None, type=click.STRING)
    @click.option("--time_budget", "-t", default=60, type=click.INT, help="Time budget in minutes")
    @click.option("--threshold", "-c", default=None, type=click.FLOAT, help="Binary cutoff")
    @click.option("--direction", "-d", default=None, type=click.STRING, help="Direction of the outcome ('high' or 'low').")
    @click.option("--parameters", "-p", default=None, type=click.STRING)
    def setup(task, input_file, output_dir, model_dir, time_budget, threshold, direction, parameters):
        echo("Reading from {0}".format(input_file))
        if task == "train":
            s = TrainSetup(
                input_file=input_file,
                output_dir=output_dir,
                time_budget=time_budget,
                threshold=threshold,
                direction=direction,
                parameters=parameters,
            )
            s.setup()
        elif task == "predict":
            s = PredictSetup(
                input_file=input_file,
                output_dir=output_dir,
                model_dir=model_dir,
                time_budget=time_budget,
            )
            s.setup()
        else:
            echo("Task must be 'train' or 'predict'", fg="red")
            sys.exit(0)
        echo("Results will be stored at {0}".format(output_dir))
        echo("Done", fg="green")
