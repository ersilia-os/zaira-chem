import click

from . import zairachem_cli
from ..echo import echo

# from ...setup.prediction import PredictSetup
# from ...descriptors.describe import Describer
# from ...estimators.pipe import EstimatorPipeline
# from ...pool.pool import Pooler
# from ...applicability.applicability import ApplicabilityEvaluator
# from ...reports.report import Reporter
# from ...finish.finisher import Finisher


def split_cmd():
    @zairachem_cli.command(help="Split input data set for cross-validation")
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option("--folds_dir", "-f", default=None, type=click.STRING)
    @click.option("--num_folds", "-n", default=None, type=click.STRING)
    def split(input_file, folds_dir, num_folds):
        if folds_dir is None:
            folds_dir = os.path.abspath(input_file))
        echo(f"Split datasets will be stored at {folds_dir}")
        echo('Hello world!')
        echo("Done", fg="green")

        s = TrainSetup(
            input_file=input_file,
            reference_file=input_file,
            output_dir=output_dir,
            parameters=parameters,
            time_budget=60,
            task=None,
            direction=None,
            threshold=None,
        )
        s.setup()
