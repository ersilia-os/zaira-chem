from .cmd import Command
from .commands import zairachem_cli


def create_cli():

    cmd = Command()

    cmd.setup()
    cmd.describe()
    cmd.estimate()
    cmd.pool()

    return zairachem_cli
