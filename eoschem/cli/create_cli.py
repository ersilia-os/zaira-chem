from .cmd import Command
from .commands import eoschem_cli


def create_cli():

    cmd = Command()

    cmd.setup()
    cmd.describe()
    cmd.fit()
    cmd.pool()
    cmd.predict()

    return eoschem_cli
