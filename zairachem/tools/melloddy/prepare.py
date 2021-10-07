import os
import pandas as pd

from . import MELLODDY_SUBFOLDER
from . import T0_FILE, T1_FILE, T2_FILE


class Prepare(object):

    def __init__(self, infile, outdir):
        self.infile = os.path.abspath(infile)
        assert os.path.isfile(infile)
        self.outdir = os.path.abspath(os.path.join(outdir, MELLODDY_SUBFOLDER))
        os.makedirs(self.outdir, exist_ok=True)

    def t0(self):
        pass # TODO

    def t1(self):
        pass # TODO

    def t2(self):
        dfi = pd.read_csv(self.infile)
        dfo = dfi[["identifier", "smiles"]]
        dfo = dfo.rename(columns={"identifier": "input_compound_id"})
        dfo.to_csv(os.path.join(self.outdir, T2_FILE), index=False)

    def prepare(self):
        self.t0()
        self.t1()
        self.t2()
