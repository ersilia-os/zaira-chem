from .cmd import Command
from .commands import zairachem_cli


def create_cli():

    cmd = Command()

    cmd.setup()
    cmd.describe()
    cmd.estimate()
    cmd.pool()
    cmd.fit()
    cmd.predict()
    cmd.plot()

    return zairachem_cli
