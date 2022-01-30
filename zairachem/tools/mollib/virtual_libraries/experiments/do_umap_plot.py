# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import argparse
import configparser
import ast
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from rdkit.Chem import AllChem as Chem
import math
import umap
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pylab as plt
import random

random.seed(16)

sys.path.append("../src/")
from python import helper as hp
from python import helper_chem as hp_chem
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description="Do umap projection")
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


def get_embedding(data):
    """Function to compute the UMAP embedding"""
    data_scaled = StandardScaler().fit_transform(data)

    embedding = umap.UMAP(
        n_neighbors=10, min_dist=0.5, metric="correlation", random_state=16
    ).fit_transform(data_scaled)

    return embedding


def combined_plot(
    embedding,
    lim_src,
    lim_tgt,
    lim_start,
    lim_end,
    e_end,
    contour_c="#444444",
    m_data="o",
    m_gen="o",
    s_data=80,
    s_e=60,
    alpha_gen=1.00,
    alpha_data=0.85,
    linewidth_gen="1.40",
    legend=False,
):

    fig, ax = plt.subplots(figsize=(14, 10))

    plt.xlim([np.min(embedding[:, 0]) - 0.5, np.max(embedding[:, 0]) + 1.5])
    plt.ylim([np.min(embedding[:, 1]) - 0.5, np.max(embedding[:, 1]) + 0.5])

    labelsize = 16
    plt.xlabel("UMAP 1", fontsize=labelsize)
    plt.ylabel("UMAP 2", fontsize=labelsize)

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    pal = FP.COLOR_PAL_CB

    plt.scatter(
        embedding[:lim_src, 0],
        embedding[:lim_src, 1],
        lw=0,
        c=pal["source"],
        label="Source space",
        alpha=alpha_data,
        s=s_data,
        marker=m_data,
    )
    plt.scatter(
        embedding[lim_src:lim_tgt, 0],
        embedding[lim_src:lim_tgt, 1],
        lw=0,
        c=pal["target"],
        label="Target space",
        alpha=alpha_data,
        s=s_data,
        marker=m_data,
    )

    plt.scatter(
        embedding[lim_tgt:lim_start, 0],
        embedding[lim_tgt:lim_start, 1],
        lw=0,
        c=pal["e_start"],
        label="Epoch 0",
        alpha=alpha_gen,
        s=s_e,
        marker=m_gen,
        edgecolors=contour_c,
        linewidth=linewidth_gen,
    )
    plt.scatter(
        embedding[lim_start:lim_end, 0],
        embedding[lim_start:lim_end, 1],
        lw=0,
        c=pal["e_end"],
        label=f"Last epoch",
        alpha=alpha_gen,
        s=s_e,
        marker=m_gen,
        edgecolors=contour_c,
        linewidth=linewidth_gen,
    )

    plt.scatter(
        embedding[lim_end:, 0],
        embedding[lim_end:, 1],
        lw=0,
        c="#5A5A5A",
        label="Target set",
        alpha=1.0,
        s=185,
        marker=m_data,
        edgecolors="k",
        linewidth="3",
    )

    if legend:
        leg = plt.legend(prop={"size": labelsize}, loc="upper right", markerscale=1.00)
        leg.get_frame().set_alpha(0.9)

    plt.setp(ax, xticks=[], yticks=[])

    return fig


def get_fp(list_of_smi):
    """Function to get fingerprint from a list of SMILES"""
    fingerprints = []
    mols = [Chem.MolFromSmiles(x) for x in list_of_smi]
    # if rdkit can't compute the fingerprint on a SMILES
    # we remove that SMILES
    idx_to_remove = []
    for idx, mol in enumerate(mols):
        try:
            fprint = Chem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=False)
            fingerprints.append(fprint)
        except:
            idx_to_remove.append(idx)

    smi_to_keep = [smi for i, smi in enumerate(list_of_smi) if i not in idx_to_remove]
    return fingerprints, smi_to_keep


def get_n_random_dataset(path, n):
    with open(f"{path}data_tr.txt", "r") as f:
        data_training = f.read().splitlines()
    with open(f"{path}data_val.txt", "r") as f:
        data_validation = f.read().splitlines()
    data = data_training + data_validation
    return random.sample(data, n)


def get_n_random(path, n):
    with open(path, "r") as f:
        data = f.read().splitlines()
    if n < len(data):
        return random.sample(data, n)
    else:
        warnings.warn("You have less data available than n")
        print(f"n = {n}")
        print(f"n data availalbe = {len(data)}")
        return data


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
        temp = float(config["EXPERIMENTS"]["temp"])

        e_end = int(config["MODEL"]["epochs"])
        if e_end < 10:
            e_end = f"0{e_end}"
        n_dataset = FP.UMAP_PLOT["n_dataset"]
        n_gen = FP.UMAP_PLOT["n_gen"]

        if verbose:
            print("\nSTART DOING UMAP PROJECTION")
        ####################################

        ####################################
        # Getting back data from source and target space
        # as well as from the pretrained model
        # Path to the data
        def get_smi_and_fp(space):
            name = config["DATA"][space]
            name = name.replace(".txt", "")
            aug = int(config["AUGMENTATION"][space])
            path = f"results/data/{name}/{min_len}_{max_len}_x{aug}/"
            list_of_smi = get_n_random_dataset(path, n_dataset)
            fp, list_of_smi = get_fp(list_of_smi)
            assert len(fp) == len(list_of_smi)

            return list_of_smi, fp

        src_smiles, src_fp = get_smi_and_fp("source_space")
        tgt_smiles, tgt_fp = get_smi_and_fp("target_space")

        aug_ft = int(config["AUGMENTATION"]["fine_tuning"])
        path_ft_smiles = f"results/data/{name_data}/{min_len}_{max_len}_x{aug_ft}/"
        ft_smiles_tr = hp.read_with_pd(f"{path_ft_smiles}data_tr.txt")
        with open(f"{path_ft_smiles}data_tr.txt", "r") as f:
            ft_smiles_tr = f.read().splitlines()
        with open(f"{path_ft_smiles}data_val.txt", "r") as f:
            ft_smiles_val = f.read().splitlines()
        ft_smiles = ft_smiles_tr + ft_smiles_val
        ft_fp, ft_smiles = get_fp(ft_smiles)
        ####################################

        ####################################
        # path to the saved novo data
        path_novo = f"results/{name_data}/novo_molecules/"

        # Path to save the UMAP plots
        save_path = f"results/{name_data}/umap/"
        os.makedirs(save_path, exist_ok=True)
        ####################################

        ####################################
        # save SMILES used here for the interative UMAP
        hp.write_in_file(f"{save_path}smiles_src.txt", src_smiles)
        hp.write_in_file(f"{save_path}smiles_tgt.txt", tgt_smiles)
        hp.write_in_file(f"{save_path}smiles_ft.txt", ft_smiles)
        ####################################

        ####################################
        # iterate over the generated data
        path_epoch_start = f"../models/molecules_start_{temp}.txt"
        e_start_smiles = get_n_random(path_epoch_start, n_gen)
        e_start_fp, e_start_smiles = get_fp(e_start_smiles)
        hp.write_in_file(f"{save_path}smiles_start_{temp}.txt", e_start_smiles)

        path_epoch_end = f"{path_novo}molecules_{e_end}_{temp}.txt"
        e_end_smiles = get_n_random(path_epoch_end, n_gen)
        e_end_fp, e_end_smiles = get_fp(e_end_smiles)
        hp.write_in_file(f"{save_path}smiles_end_{temp}.txt", e_end_smiles)

        path_projection = f"{save_path}umap_projection_{temp}.npy"
        if not os.path.isfile(path_projection):
            src_fp = np.array(src_fp)
            tgt_fp = np.array(tgt_fp)
            e_start_fp = np.array(e_start_fp)
            e_end_fp = np.array(e_end_fp)
            ft_fp = np.array(ft_fp)

            all_data = np.concatenate(
                [src_fp, tgt_fp, e_start_fp, e_end_fp, ft_fp], axis=0
            )
            if verbose:
                print(f"all_data shape: {all_data.shape}")

            embedding = get_embedding(all_data)

            assert embedding.shape[0] == all_data.shape[0]
            assert embedding.shape[1] != all_data.shape[1]

            np.save(path_projection, embedding)
        else:
            embedding = np.load(path_projection)

        lim_src = len(src_fp)
        lim_tgt = lim_src + len(tgt_fp)
        lim_start = lim_tgt + len(e_start_fp)
        lim_end = lim_start + len(e_end_fp)

        common_emb = embedding[:lim_end, :]
        ft_emb = embedding[lim_end:, :]
        if verbose:
            print(
                f"Embedding shapes, common: {common_emb.shape}, fine-tuning: {ft_emb.shape}"
            )

        fig_S = combined_plot(
            np.concatenate([common_emb, ft_emb]),
            lim_src,
            lim_tgt,
            lim_start,
            lim_end,
            e_end,
            legend=True,
        )
        plt.savefig(f"{save_path}umap_{temp}.png", dpi=150)

        end = time.time()
        if verbose:
            print(f"UMAP PROJECTION DONE in {end - start:.04} seconds")

    else:
        print("Defaut paremeters not used; UMAP not done.")
    ####################################
