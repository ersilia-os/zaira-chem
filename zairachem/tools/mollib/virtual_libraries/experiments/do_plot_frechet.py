# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import argparse
import configparser
import ast
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append("../src/")
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description="Do Frechet distance plot")
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


def do_plot(dict_src, dict_tgt, save_path, dashes=None):

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # get back the FCD and epoch from dict
    x, y_src = zip(*sorted(dict_src.items()))
    _, y_tgt = zip(*sorted(dict_tgt.items()))
    df = pd.DataFrame(np.c_[y_src, y_tgt], index=x)

    # line plot
    if dashes:
        ax = sns.lineplot(data=df, dashes=dashes)
    else:
        ax = sns.lineplot(data=df)

    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    tick_font_sz = FP.PAPER_FONT["tick_font_sz"]
    label_font_sz = FP.PAPER_FONT["label_font_sz"]
    legend_sz = FP.PAPER_FONT["legend_sz"]

    plt.xlabel("Epoch", fontsize=label_font_sz)
    plt.ylabel("FCD", fontsize=label_font_sz)

    ax.set_ylim(0.0, 100)

    plt.yticks(fontsize=tick_font_sz)
    plt.xticks(fontsize=tick_font_sz)

    plt.legend(
        loc="upper right", labels=["To source space", "To target space"], frameon=False
    )
    plt.setp(ax.get_legend().get_texts(), fontsize=legend_sz)

    plt.savefig(save_path)
    plt.close(fig)


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

    temp = float(config["EXPERIMENTS"]["temp"])

    if verbose:
        print("\nSTART FRECHET PLOT")
    ####################################

    ####################################
    # Path to the FCD data
    path_fcd = f"results/{name_data}/frechet/"

    # Path to save the novo analysis
    save_path = f"results/{name_data}/plot_frechet/"
    os.makedirs(save_path, exist_ok=True)
    ####################################

    ####################################
    # and do the plot
    flatui_alone = ["#000000", "#000000"]
    sns.set_palette(flatui_alone)
    dashe_space = 25
    dashe_len = 12.5
    dashes = None

    dict_src = {}
    dict_tgt = {}

    # start plotting
    for filename in os.listdir(path_fcd):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")
            data = hp.load_obj(f"{path_fcd}{name}")
            epoch = int(name.split("_")[1])
            te = name.split("_")[2]
            if float(temp) == float(te):
                dict_src[epoch] = data["f_dist_src"]
                dict_tgt[epoch] = data["f_dist_tgt"]

    do_plot(dict_src, dict_tgt, f"{save_path}frechet_distance_{te}.png", dashes=dashes)

    end = time.time()
    if verbose:
        print(f"FRECHET PLOT DONE in {end - start:.04} seconds")
    ####################################
