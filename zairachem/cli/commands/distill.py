import click
from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo
from ...distill.distill import Distiller


def distill_cmd():
    @zairachem_cli.command(help="Distill model to ONNX format with Olinda pipeline")
    @click.option("--model_dir", "-m", default=None, type=click.STRING)
    @click.option("--output_path", "-o", default=None, type=click.STRING)
    def distill(model_dir, output_path):
        echo("Distilling model with Olinda")
        distill = Distiller(zaira_path=model_dir, onnx_output_path=output_path)
        distill.run()
        echo("Done", fg="green")
