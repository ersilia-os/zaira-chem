import click

from . import zairachem_cli
from ..echo import echo

from ...setup.training import TrainSetup
from ...descriptors.describe import Describer
from ...estimators.pipe import EstimatorPipeline
from ...pool.pool import Pooler
from ...applicability.applicability import ApplicabilityEvaluator
from ...reports.report import Reporter
from ...finish.finisher import Finisher


def fit_cmd():
    @zairachem_cli.command(help="Fit an ML-based QSAR model")
    @click.option("--input_file", "-i", type=click.STRING)
    @click.option(
        "--reference_file",
        "-r",
        default=None,
        type=click.STRING,
        help="Reference file containing a relatively large set of molecules (at least 1000). This parameter is currently not used...",
    )
    @click.option(
        "--model_dir",
        "-m",
        default=None,
        type=click.STRING,
        help="Directory where the model should be stored.",
    )
    @click.option(
        "--time_budget",
        "-b",
        default=120,
        type=click.INT,
        help="Time budget in minutes. This option is currently not used.",
    )
    @click.option(
        "--task",
        "-t",
        default=None,
        type=click.STRING,
        help="Type of task: 'classification' or 'regression'. If not specified, ZairaChem guesses the task based on the data and the user input (i.e. if cutoff is specified, it assumes classification)",
    )
    @click.option(
        "--cutoff",
        "-c",
        default=None,
        type=click.FLOAT,
        help="Cutoff to binarize data, i.e. to separate actives and inactives. By convention, actives = 1 and inactives = 0, check 'direction'.",
    )
    @click.option(
        "--direction",
        "-d",
        default=None,
        type=click.STRING,
        help="Direction of the actives: 'high' means that high values are actives, 'low' means that low values are actives.",
    )
    @click.option(
        "--parameters",
        "-p",
        default=None,
        type=click.STRING,
        help="Path to parameters file in JSON format.",
    )
    @click.option(
        "--clean",
        is_flag=True,
        show_default=True,
        default=False,
        help="Clean directory at the end of the pipeline. Only precalculated descriptors are removed.",
    )
    def fit(
        input_file,
        reference_file,
        model_dir,
        time_budget,
        task,
        cutoff,
        direction,
        parameters,
        clean,
    ):
        output_dir = model_dir
        threshold = cutoff
        echo("Results will be stored at {0}".format(output_dir))
        s = TrainSetup(
            input_file=input_file,
            reference_file=reference_file,
            output_dir=output_dir,
            parameters=parameters,
            time_budget=time_budget,
            task=task,
            direction=direction,
            threshold=threshold,
        )
        if s.is_done():
            echo("Model has already been trained. Skipping")
            return
        s.setup()
        d = Describer(path=output_dir)
        d.run()
        e = EstimatorPipeline(path=output_dir)
        e.run()
        p = Pooler(path=output_dir)
        p.run()
        a = ApplicabilityEvaluator(path=output_dir)
        a.run()
        r = Reporter(path=output_dir)
        r.run()
        f = Finisher(path=output_dir, clean=clean, flush=False)
        f.run()
        echo("Done", fg="green")
