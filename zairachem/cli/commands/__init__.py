import click
from ... import __version__
from ... import logger


@click.group(cls=click.Group)
@click.version_option(version=__version__)
@click.option(
    "--verbose",
    default=False,
    is_flag=True,
    help="Show logging on terminal when running commands.",
)
def zairachem_cli(verbose):
    """
    Ersilia Chemistry AutoML CLI
    """
    if verbose:
        logger.set_verbosity(1)
    else:
        logger.set_verbosity(0)
    logger.debug("Starting CLI")
