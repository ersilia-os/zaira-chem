# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import warnings
import argparse
import configparser
import ast
import numpy as np
from math import log
from rdkit import Chem
from rdkit import rdBase

rdBase.DisableLog("rdApp.*")
from rdkit.Chem import Draw
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.models import load_model

sys.path.append("../src/")
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description="SMILES generation")
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument(
    "-m", "--model_path", type=str, help="Path to a pretrained model", required=True
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


def int_to_smile(array, indices_token, pad_char):
    """
    From an array of int, return a list of
    molecules in string smile format
    Note: remove the padding char
    """
    all_mols = []
    for seq in array:
        new_mol = [indices_token[str(int(x))] for x in seq]
        all_mols.append("".join(new_mol).replace(pad_char, ""))
    return all_mols


def one_hot_encode(token_lists, n_chars):

    output = np.zeros((len(token_lists), len(token_lists[0]), n_chars))
    for i, token_list in enumerate(token_lists):
        for j, token in enumerate(token_list):
            output[i, j, int(token)] = 1
    return output


def sample(model, temp, start_char, end_char, max_len, indices_token, token_indices):

    n_chars = len(indices_token)

    seed_token = [token_indices[start_char]]
    generated = indices_token[str(seed_token[0])]

    while generated[-1] != end_char and len(generated) < max_len:
        x_seed = one_hot_encode([seed_token], n_chars)
        full_preds = model.predict(x_seed, verbose=0)[0]
        logits = full_preds[-1]

        probas, next_char_ind = get_token_proba(logits, temp)

        next_char = indices_token[str(next_char_ind)]
        generated += next_char
        seed_token += [next_char_ind]

    return generated


def get_token_proba(preds, temp):

    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temp
    exp_preds = np.exp(preds)

    probas = exp_preds / np.sum(exp_preds)
    char_ind = np.argmax(np.random.multinomial(1, probas, 1))

    return probas, char_ind


def softmax(preds):
    return np.exp(preds) / np.sum(np.exp(preds))


if __name__ == "__main__":

    start = time.time()

    ####################################
    # get back parameters
    args = vars(parser.parse_args())

    verbose = args["verbose"]
    filename = args["filename"]
    model_path = args["model_path"]
    name_data = filename.split("/")[-1].replace(".txt", "")
    config = configparser.ConfigParser()
    config.read("parameters.ini")
    ####################################

    ####################################
    # path to save data
    save_path = f"results/{name_data}/generated_data/"
    os.makedirs(save_path, exist_ok=True)

    # path to checkpoints
    dir_ckpts = f"results/{name_data}/models/"
    ####################################

    ####################################
    # Parameters to sample novo smiles
    temp = float(config["EXPERIMENTS"]["temp"])
    n_sample = int(config["EXPERIMENTS"]["n_sample"])
    if n_sample > 5000:
        warnings.warn("You will sample more than 5000 SMILES; this will take a while")

    max_len = int(config["PROCESSING"]["max_len"])
    pad_char = FP.PROCESSING_FIXED["pad_char"]
    start_char = FP.PROCESSING_FIXED["start_char"]
    end_char = FP.PROCESSING_FIXED["end_char"]
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    ####################################

    ####################################
    # start the sampling of new SMILES
    epoch = model_path.split("/")[-1].replace(".h5", "")
    if verbose:
        print(f"Sampling from model saved at epoch {epoch}")

    model = load_model(model_path)

    generated_smi = []
    counter = 0
    start_sampling = time.time()
    for n in range(n_sample):
        generated_smi.append(
            sample(
                model,
                temp,
                start_char,
                end_char,
                max_len + 1,
                indices_token,
                token_indices,
            )
        )

        # From 100 molecules to sample,
        # we indicate the current status
        # to the user
        if n_sample >= 100:
            if len(generated_smi) % int(0.1 * n_sample) == 0:
                counter += 10
                delta_time = time.time() - start_sampling
                start_sampling = start_sampling + delta_time
                print(
                    f"Status for model from epoch {epoch}: {counter}% of the molecules sampled in {delta_time:.2f} seconds"
                )

    hp.save_obj(generated_smi, f"{save_path}{epoch}_{temp}")

    end = time.time()
    if verbose:
        print(f"SAMPLING DONE for model from epoch {epoch} in {end-start:.2f} seconds")
    ####################################
