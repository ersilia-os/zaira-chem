# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import argparse
import configparser
import ast
from rdkit import Chem
from rdkit import rdBase

rdBase.DisableLog("rdApp.*")

sys.path.append("../src/")
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description="Run novo analysis")
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


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
    mode = config["EXPERIMENTS"]["mode"]
    pad_char = FP.PROCESSING_FIXED["pad_char"]
    start_char = FP.PROCESSING_FIXED["start_char"]
    end_char = FP.PROCESSING_FIXED["end_char"]
    min_len = int(config["PROCESSING"]["min_len"])
    max_len = int(config["PROCESSING"]["max_len"])

    # experiment parameters depending on the mode
    augmentation = int(config["AUGMENTATION"][mode])

    if verbose:
        print("\nSTART NOVO ANALYSIS")
    ####################################

    ####################################
    # Path to the generated data
    path_gen = f"results/{name_data}/generated_data/"

    # Path to save the novo analysis
    save_path = f"results/{name_data}/novo_molecules/"
    os.makedirs(save_path, exist_ok=True)
    ####################################

    ####################################
    # Load the original dataset for the novo analysis
    # note that we load the train-valid split done on the
    # data without augmentation -> we want to compare to the
    # canonical SMILES
    dir_split_data = f"results/data/{name_data}/{min_len}_{max_len}_x{augmentation}/"

    with open(f"{dir_split_data}data_tr.txt", "r") as f:
        data_training = f.readlines()
    with open(f"{dir_split_data}data_val.txt", "r") as f:
        data_validation = f.readlines()
    ####################################

    ####################################
    # Start iterating over the files
    t0 = time.time()
    for filename in os.listdir(path_gen):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")
            data = hp.load_obj(path_gen + name)

            valids = []
            n_valid = 0

            for gen_smile in data:
                if len(gen_smile) != 0 and isinstance(gen_smile, str):
                    gen_smile = gen_smile.replace(pad_char, "")
                    gen_smile = gen_smile.replace(end_char, "")
                    gen_smile = gen_smile.replace(start_char, "")

                    mol = Chem.MolFromSmiles(gen_smile)
                    if mol is not None:
                        cans = Chem.MolToSmiles(
                            mol, isomericSmiles=True, canonical=True
                        )
                        if len(cans) >= 1:
                            n_valid += 1
                            valids.append(cans)

            if n_valid != 0:
                # Now let's pruned our valid guys
                unique_set = set(valids)
                n_unique = len(unique_set)
                novo_tr = list(unique_set - set(data_training))
                n_novo_tr = len(novo_tr)
                novo_val = list(unique_set - set(data_validation))
                n_novo_val = len(novo_val)
                novo_analysis = {
                    "n_valid": n_valid,
                    "n_unique": n_unique,
                    "n_novo_tr": n_novo_tr,
                    "n_novo_val": n_novo_val,
                    "novo_tr": novo_tr,
                }

                # we save the novo molecules also as .txt
                novo_name = f"{save_path}molecules_{name}"
                with open(f"{novo_name}.txt", "w+") as f:
                    for item in novo_tr:
                        f.write("%s\n" % item)

                hp.save_obj(novo_analysis, novo_name)

                if verbose:
                    print(f"sampling analysis for {name} done")
            else:
                print(f"There are n {n_valid} valids SMILES for {name}")

    end = time.time()
    if verbose:
        print(f"NOVO ANALYSIS DONE in {end - start:.04} seconds")
    ####################################
