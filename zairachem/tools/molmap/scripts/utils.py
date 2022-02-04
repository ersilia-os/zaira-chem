import sys
import os

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "../bidd-molmap/"))

import molmap

metric = "cosine"


def descriptors_molmap():
    return molmap.MolMap(ftype="descriptor", metric="cosine").load(
        filename=os.path.join(root, "../data/descriptor.mp")
    )


def fingerprints_molmap():
    bitsinfo = molmap.feature.fingerprint.Extraction().bitsinfo
    flist = bitsinfo[
        bitsinfo.Subtypes.isin(["MACCSFP", "PharmacoErGFP", "PubChemFP"])
    ].IDs.tolist()
    return molmap.MolMap(ftype="fingerprint", metric=metric, flist=flist).load(
        filename="../data/fingerprint.mp"
    )
