# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import numpy as np
import argparse
import configparser
import ast
import warnings
import random

random.seed(16)

sys.path.append("../src/python/")
import helper as hp
import fixed_parameters as FP
from fcd import FCD as FCD

parser = argparse.ArgumentParser(description="Compute Frechet distance")
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


def get_frechet_dist(data, generated):
    """
    Function to get back the Frechet ChemNet Distance

    Parameters:
    - data (list of SMILES string): randomly picked SMILES string from a dataset.
    - generated (list of SMILES string): randomly picked generated SMILES string.

    return: FCD value between data and generated
    """

    n_data = min([len(data), len(generated)])
    print(f"{n_data} molecules were taken to compute FCD (-> min(data, generated))")

    mols_1_act = FCD.get_predictions(data[:n_data])
    mols_2_act = FCD.get_predictions(generated[:n_data])

    fcd = FCD.calculate_frechet_distance(
        mu1=np.mean(mols_1_act, axis=0),
        mu2=np.mean(mols_2_act, axis=0),
        sigma1=np.cov(mols_1_act.T),
        sigma2=np.cov(mols_2_act.T),
    )

    return fcd


def get_back_data(path):
    with open(f"{path}data_tr.txt", "r") as f:
        data_training = f.readlines()
    with open(f"{path}data_val.txt", "r") as f:
        data_validation = f.readlines()

    return data_training + data_validation


def get_n_random(data, n, name):
    if n < len(data):
        return random.sample(data, n)
    else:
        warnings.warn("You have less data available than n")
        print(f"n = {n}")
        print(f"n data availalbe = {len(data)} for {name}")
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

    # get back the experiment parameters
    min_len = int(config["PROCESSING"]["min_len"])
    max_len = int(config["PROCESSING"]["max_len"])
    N_fcd = FP.FRECHET["n_data"]

    if verbose:
        print("\nSTART COMPUTING FRECHET DISTANCE")
    ####################################

    ####################################
    # Path to the generated data
    path_gen = f"results/{name_data}/novo_molecules/"

    # Path to save the frechet distance
    save_path = f"results/{name_data}/frechet/"
    os.makedirs(save_path, exist_ok=True)
    ####################################

    ####################################
    # Path to the dataset to calculate the
    # frechet distance
    def get_data(space):
        name = config["DATA"][space]
        name = name.replace(".txt", "")
        aug = int(config["AUGMENTATION"][space])
        path = f"results/data/{name}/{min_len}_{max_len}_x{aug}/"
        data = get_back_data(path)
        data = get_n_random(data, N_fcd, name)

        return data

    src_data = get_data("source_space")
    tgt_data = get_data("target_space")
    ####################################

    ####################################
    # start iterating over the files
    for filename in os.listdir(path_gen):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")
            epoch = int(name.split("_")[1])
            temp = float(name.split("_")[2])

            data = hp.load_obj(f"{path_gen}{name}")
            novo = data["novo_tr"]
            novo = get_n_random(novo, N_fcd, name)

            f_dist_src = get_frechet_dist(src_data, novo)
            f_dist_tgt = get_frechet_dist(tgt_data, novo)

            frechet_dist = {"f_dist_src": f_dist_src, "f_dist_tgt": f_dist_tgt}

            hp.save_obj(frechet_dist, f"{save_path}frechet_{epoch}_{temp}")
            if verbose:
                print(
                    f"e: {epoch}, t: {temp}, FCD to src: {f_dist_src:.03}, FCD to tgt: {f_dist_tgt:.03}"
                )

    end = time.time()
    if verbose:
        print(f"FRECHET DISTANCE DONE in {end - start:.04} seconds")
    ####################################
