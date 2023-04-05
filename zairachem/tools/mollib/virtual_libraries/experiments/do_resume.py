# Copyright (c) 2019 ETH Zurich

import os, sys
import argparse
import configparser
import ast
import re
import glob
import time
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

sys.path.append("../src/")
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(
    description="Join all pictures into one to create an overview resume"
)
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


def draw_at_temp(temp, name_data, epochs_to_plot, save_path, descriptor):
    """
    Function to create the resume picture
    of the experiment.

    temp: temperature
    path_data: path to data
    """
    root = f"results/{name_data}/"

    frechet = f"{root}plot_frechet/frechet_distance_{temp}.png"
    desc = f"{root}plot_descriptor/{descriptor}_{temp}.png"
    umap = f"{root}umap/umap_{temp}.png"

    files = [frechet, desc, umap]

    for e in epochs_to_plot:
        scaffolds = f"{root}drawing_scaffolds/scaffolds/{temp}/{e}_top_5.png"
        files.append(scaffolds)

    result = Image.new("RGB", (2480, 3508), color="white")
    header = 0

    for index, file in enumerate(files):
        path = os.path.expanduser(file)
        img = Image.open(path)
        w, h = img.size
        if index == 0:
            x = 63
            y = 0 + header
        elif index == 1:
            x = 1300
            y = 50 + header
        elif index == 2:
            crop_big = 200
            crop_small = 50
            img = img.crop((crop_big, crop_big, w - crop_small, h - crop_small))
            w, h = img.size
            x = 155
            y = 920 + header
            ratio = 0.80
            img.thumbnail((w * ratio, h * ratio), Image.ANTIALIAS)
        elif index == 3:
            w, h = img.size
            x = 90
            y = 1900 + header
            ratio = 1.70
            img = img.resize((int(w * ratio), int(h * ratio)), Image.ANTIALIAS)
        elif index == 4:
            w, h = img.size
            x = 90
            y = 2300 + header
            ratio = 1.70
            img = img.resize((int(w * ratio), int(h * ratio)), Image.ANTIALIAS)
        elif index == 5:
            w, h = img.size
            x = 90
            y = 2700 + header
            ratio = 1.70
            img = img.resize((int(w * ratio), int(h * ratio)), Image.ANTIALIAS)
        elif index == 6:
            w, h = img.size
            x = 90
            y = 3100 + header
            ratio = 1.70
            img = img.resize((int(w * ratio), int(h * ratio)), Image.ANTIALIAS)

        # we need the updated sizes
        w, h = img.size
        result.paste(img, (x, y, x + w, y + h))

        # add a,b,c,d
        draw = ImageDraw.Draw(result)
        font = ImageFont.truetype("../src/python/fonts/Arial.ttf", 50)
        draw.text((60, 25), "a", (0, 0, 0), font=font)
        draw.text((1250, 25), "b", (0, 0, 0), font=font)
        draw.text((60, 920), "c", (0, 0, 0), font=font)
        draw.text((60, 1880), "d", (0, 0, 0), font=font)
        # legend
        draw = ImageDraw.Draw(result)

        font = ImageFont.truetype("../src/python/fonts/Arial.ttf", 35)
        txtzero = "Legend\n"
        txta = "a, Fréchet ChemNet Distance\nto source and target space\n"
        if desc_to_plot == "FractionCSP3":
            name_ = "Fsp3"
        else:
            name_ = desc_to_plot
        txtb = f"b, Evolution of the {name_}\n"
        txtc = "c, UMAP plot of molecules\n"
        txtd = "d, Five most frequent scaffolds"
        all_txt = txtzero + txta + txtb + txtc + txtd

        draw.text((1600, 1200), all_txt, (0, 0, 0), font=font)

    result.save(f"{save_path}resume_{temp}.jpg")


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
        temp = float(config["EXPERIMENTS"]["temp"])
        desc_to_plot = FP.DESCRIPTORS["names"]
        desc_to_plot = re.search(r"\((.*?)\)", desc_to_plot).group(1)

        if verbose:
            print("\nSTART RESUME")
        ####################################

        ####################################
        # path to save the drawing
        save_path = f"results/{name_data}/resume/"
        os.makedirs(save_path, exist_ok=True)
        ####################################

        ####################################
        # Start doing the resume file
        # We get back the epoch sampled from the saved models
        all_models = glob.glob(f"results/{name_data}/models/*.h5")
        epochs_to_plot = [x.split("/")[-1].replace(".h5", "") for x in all_models]
        draw_at_temp(temp, name_data, epochs_to_plot, save_path, desc_to_plot)

        end = time.time()
        if verbose:
            print(f"RESUME DONE in {end - start:.04} seconds")
    else:
        print("Defaut paremeters not used; resume not done.")
    ####################################
