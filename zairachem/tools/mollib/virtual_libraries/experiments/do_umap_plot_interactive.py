# Copyright (c) 2019 ETH Zurich

import os, sys
import argparse
import configparser
import ast
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
import altair as alt
import numpy as np
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pylab as plt
import seaborn as sns

sys.path.append("../src/")
from python import helper as hp
from python import helper_chem as hp_chem
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description="Do umap interactive plot")
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


def get_dataframe(list_of_smi, embedding, time):
    """
    rule of 5 taken from: https://squonk.it/docs/cells/Lipinski%20filter%20(RDKit)/
    """
    mols = [Chem.MolFromSmiles(smi) for smi in list_of_smi]
    assert len(mols) == embedding.shape[0]

    FractionCSP3 = []
    mol_weights = []
    LogP = []
    H_donor = []
    H_acceptor = []

    # We have to do this cubersome for loop
    # because rdkit is sometimes unhappy about
    # some SMILES and is throwing weird error
    # when computing certain properties.
    # If this happens here, we return a None value
    for mol in mols:
        try:
            fsp3 = round(Descriptors.FractionCSP3(mol), 4)
        except:
            fsp3 = None
        FractionCSP3.append(fsp3)
        try:
            mweight = round(Descriptors.ExactMolWt(mol), 2)
        except:
            mweight = None
        mol_weights.append(mweight)
        try:
            lp = round(Descriptors.MolLogP(mol), 2)
        except:
            lp = None
        LogP.append(lp)
        try:
            hdonor = round(Lipinski.NumHDonors(mol), 2)
        except:
            hdonor = None
        H_donor.append(hdonor)
        try:
            hacceptor = round(Lipinski.NumHAcceptors(mol), 2)
        except:
            hacceptor = None
        H_acceptor.append(hacceptor)

    # rule of 5
    limit_mw = 500
    limit_LogP = 5.0
    limit_H_donor = 5
    limit_H_acceptor = 10

    lipinski = []
    for m, l, hd, ha in zip(mol_weights, LogP, H_donor, H_acceptor):
        if m is None or l is None or hd is None or ha is None:
            lipinski.append("unknown")
        elif (
            m <= limit_mw
            and l <= limit_LogP
            and hd <= limit_H_donor
            and ha <= limit_H_acceptor
        ):
            lipinski.append("respected")
        else:
            lipinski.append("not respected")

    moldf = {}
    moldf["Cluster"] = [f"{time}"] * embedding.shape[0]
    moldf["UMAP1"] = embedding[:, 0]
    moldf["UMAP2"] = embedding[:, 1]
    moldf["SMILES"] = list_of_smi
    moldf["url"] = ["http://molview.org/?q=" + x for x in list_of_smi]
    moldf["Molecular weights"] = mol_weights
    moldf["Fraction Csp3"] = FractionCSP3
    moldf["LogP"] = LogP
    moldf["H-bond donor count"] = H_donor
    moldf["H-bond acceptor count"] = H_acceptor

    moldf["Rule of 5"] = lipinski

    df = pd.DataFrame.from_dict(moldf)

    return df


def do_interactive_chart(df, save_path):
    chart = (
        alt.Chart(df)
        .transform_calculate()
        .mark_point(filled=True, size=60)
        .encode(
            x="UMAP1",
            y="UMAP2",
            color=alt.Color("Cluster", scale=alt.Scale(scheme="tableau20")),
            tooltip=[
                "SMILES",
                "Molecular weights",
                "Fraction Csp3",
                "LogP",
                "H-bond donor count",
                "H-bond acceptor count",
                "Rule of 5",
            ],
            href="url",
        )
        .interactive()
        .properties(width=800, height=600)
        .configure_axis(grid=False, ticks=False)
        .configure_view(strokeWidth=0)
        .configure_header(labelFontSize=16)
    )

    chart.save(f"{save_path}.html")


if __name__ == "__main__":
    start = time.time()

    ####################################
    # get back parameters
    args = vars(parser.parse_args())

    verbose = args["verbose"]
    filename = args["filename"]
    name_data = filename.split("/")[-1].replace(".txt", "")
    config = configparser.ConfigParser()
    config.read("parameters.ini")

    # We do the UMAP only if the default parameters
    # were run, i.e. 40 epochs and models saved
    # every 10 epochs (period = 10)
    check_epoch = int(config["MODEL"]["epochs"])
    check_period = int(config["MODEL"]["period"])

    if check_epoch == 40 and check_period == 10:
        min_len = int(config["PROCESSING"]["min_len"])
        max_len = int(config["PROCESSING"]["max_len"])
        mode = config["EXPERIMENTS"]["mode"]
        e_end = int(config["MODEL"]["epochs"])
        if e_end < 10:
            e_end = f"0{e_end}"
        n_dataset = FP.UMAP_PLOT["n_dataset"]
        temp = float(config["EXPERIMENTS"]["temp"])

        if verbose:
            print("\nSTART DOING INTERACTIVE UMAP PROJECTION")
        ####################################

        ####################################
        # path to the saved UMAP embedding
        # and to save the interative UMAP
        path_umap = f"results/{name_data}/umap/"
        ####################################

        ####################################
        # Do the plot
        path_projection = f"{path_umap}umap_projection_{temp}.npy"
        embedding = np.load(path_projection)

        with open(f"{path_umap}smiles_src.txt", "r") as f:
            smiles_src = f.read().splitlines()
        with open(f"{path_umap}smiles_tgt.txt", "r") as f:
            smiles_tgt = f.read().splitlines()
        with open(f"{path_umap}smiles_start_{temp}.txt", "r") as f:
            smiles_start = f.read().splitlines()
        with open(f"{path_umap}smiles_end_{temp}.txt", "r") as f:
            smiles_end = f.read().splitlines()
        with open(f"{path_umap}smiles_ft.txt", "r") as f:
            smiles_ft = f.read().splitlines()

        lim_src = len(smiles_src)
        lim_tgt = lim_src + len(smiles_tgt)
        lim_start = lim_tgt + len(smiles_start)
        lim_end = lim_start + len(smiles_end)

        # get separate information
        df_src = get_dataframe(smiles_src, embedding[:lim_src, :], "Source space")

        df_tgt = get_dataframe(
            smiles_tgt, embedding[lim_src:lim_tgt, :], "Target space"
        )

        df_start = get_dataframe(
            smiles_start, embedding[lim_tgt:lim_start, :], "First epoch"
        )

        df_end = get_dataframe(
            smiles_end, embedding[lim_start:lim_end, :], "Last epoch"
        )

        df_ft = get_dataframe(smiles_ft, embedding[lim_end:, :], "Target set")

        # concate dataframe
        frames = [df_src, df_tgt, df_start, df_end, df_ft]
        frames_concat = pd.concat(frames)

        # plot
        do_interactive_chart(frames_concat, f"{path_umap}interative_umap")

        end = time.time()
        if verbose:
            print(f"INTERACTIVE UMAP PROJECTION DONE in {end - start:.04} seconds")

    else:
        print("Defaut paremeters not used; interactive UMAP not done.")
    ####################################
