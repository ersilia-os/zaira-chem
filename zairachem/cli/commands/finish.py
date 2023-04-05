import click

from ... import create_session_symlink

from . import zairachem_cli
from ..echo import echo
from ...finish.finisher import Finisher


def finish_cmd():
    @zairachem_cli.command(help="Finish pipeline")
    @click.option("--dir", "-d", type=click.STRING)
    @click.option(
        "--clean",
        is_flag=True,
        show_default=True,
        default=False,
        help="Clean directory at the end of the pipeline. Only precalculated descriptors are removed.",
    )
    @click.option(
        "--flush",
        is_flag=True,
        show_default=True,
        default=False,
        help="Flush directory at the end of the pipeline. Only data, results and reports are kept. Use with caution: the original trained model will be flushed too.",
    )
    @click.option(
        "--anonymize",
        is_flag=True,
        show_default=True,
        default=False,
        help="Remove all information about training set, including smiles, physchem propertie and descriptors",
    )
    def finish(dir, flush, clean, anonymize):
        create_session_symlink(dir)
        echo("Finishing".format(dir))
        r = Finisher(path=dir, flush=flush, clean=clean, anonymize=anonymize)
        r.run()
        echo("Done", fg="green")
