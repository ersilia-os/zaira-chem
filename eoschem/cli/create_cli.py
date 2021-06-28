from .cmd import Command
from .commands import eoschem_cli

def create_cli():

    cmd = Command()

    cmd.fit()
    cmd.predict()

    return eoschem_cli
