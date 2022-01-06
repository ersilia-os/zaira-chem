import sys
import os

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "../bidd-molmap/"))

from molmap import MolMap
from molmap import feature

# Descriptor
mp1_name = os.path.join(root, "../data/descriptor.mp")
mp1 = MolMap(ftype="descriptor", metric="cosine")
mp1.fit(verbose=0, method="umap", min_dist=0.1, n_neighbors=15)
mp1.save(mp1_name)

# Fingerprint
mp2_name = os.path.join(root, "../data/fingerprint.mp")
bitsinfo = feature.fingerprint.Extraction().bitsinfo
flist = bitsinfo[bitsinfo.Subtypes.isin(["PubChemFP"])].IDs.tolist()
mp2 = MolMap(ftype="fingerprint", fmap_type="scatter", flist=flist)
mp2.fit(method="umap", min_dist=0.1, n_neighbors=15, verbose=0)
mp2.save(mp2_name)
