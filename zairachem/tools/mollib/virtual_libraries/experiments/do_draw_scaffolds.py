# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import math
import argparse
import configparser
import ast
import collections
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import inflect
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.append("../src/")
from python import helper as hp

parser = argparse.ArgumentParser(
    description="Draw scaffolds from their smile representation"
)
parser.add_argument(
    "-fn",
    "--filename",
    type=str,
    help="Path to the fine-tuning txt file",
    required=True,
)
parser.add_argument(
    "-s",
    "--scaffolds_type",
    type=str,
    help="Type of the scaffolds. generic_scaffolds or scaffolds.",
    required=True,
)
parser.add_argument("-v", "--verbose", type=bool, help="Verbose", required=True)


def sdi(data, scaled=True):
    """
    Function to compute the scaled shannon
    entropy.
    if there is only one type in the dataset,
    then sdi = 0
    Maxium diversity is reach with 1
    if the scaled factor is used
    """

    def p(n, N):
        """Relative abundance"""
        if n is 0:
            return 0
        else:
            return (float(n) / N) * math.log2(float(n) / N)

    N = sum(data.values())

    if scaled:
        N_scaffolds = len(data)
        return -sum(p(n, N) for n in data.values() if n is not 0) / math.log2(
            N_scaffolds
        )
    else:
        return -sum(p(n, N) for n in data.values() if n is not 0)


def add_txt_on_img(img, text, save_name):
    """Function to add text on an image"""
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("../src/python/fonts/Arial.ttf", 18)
    draw.text((10, 10), text, (0, 0, 0), font=font)
    img.save(save_name)


def draw_at_temp(temp, path_data, save_path):
    """
    Function to draw scaffolds with rdkit

    temp: temperature
    path_data: path to data
    save_path: path to save pictures
    """
    for filename in os.listdir(path_data):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")
            epoch = name.split("_")[1]
            te = name.split("_")[2]

            if float(temp) == float(te) and "scaf" in name:
                data_ = hp.load_obj(path_data + name)

                for name_data, data in data_.items():
                    # some moleculres are put as a list with the string
                    # error; we remove them for drawing
                    # note that they are very rare
                    data = [x for x in data if type(x) is str]
                    counter = collections.Counter(data)

                    figure_top_common_combined = 5
                    top_common_combined = 20
                    to_plot = [figure_top_common_combined, top_common_combined]

                    for top_common in to_plot:
                        common = counter.most_common(top_common)

                        # all diff scaffolds we have
                        total = sum(counter.values())

                        mols = [Chem.MolFromSmiles(x[0]) for x in common]
                        repet = [f"{100*x[1]/total:.2f}%" for x in common]

                        # print a common plot of all those guys
                        common_top = Draw.MolsToGridImage(
                            mols, molsPerRow=5, subImgSize=(242, 242), legends=repet
                        )

                        save_dir_common = f"{save_path}{name_data}/{te}/"
                        os.makedirs(save_dir_common, exist_ok=True)
                        save_filename = f"{save_dir_common}{epoch}_top_{top_common}.png"
                        common_top.save(save_filename)

                        # add SSE
                        sse = sdi(dict(common), scaled=True)
                        img = Image.open(save_filename)
                        number_t_write = len(common)
                        if number_t_write < 10:
                            p = inflect.engine()
                            number_t_write = p.number_to_words(number_t_write).title()
                        text = f"{number_t_write} most common scaffolds at epoch {epoch} (SSE = {sse:.02}):"
                        add_txt_on_img(img, text, save_filename)


if __name__ == "__main__":
    start = time.time()

    ####################################
    # get back parameters
    args = vars(parser.parse_args())

    verbose = args["verbose"]
    scaffolds_type = args["scaffolds_type"]
    filename = args["filename"]
    name_data = filename.split("/")[-1].replace(".txt", "")
    config = configparser.ConfigParser()
    config.read("parameters.ini")

    temp = float(config["EXPERIMENTS"]["temp"])

    if verbose:
        print(f"\nSTART DRAWING SCAFFOLDS FOR {scaffolds_type}")
    ####################################

    ####################################
    # path to the saved scaffolds
    path_scaf = f"results/{name_data}/analysis/"

    # path to save the drawing
    save_path = f"results/{name_data}/drawing_scaffolds/"
    ####################################

    ####################################
    # start drawing
    draw_at_temp(temp, path_scaf, save_path)

    end = time.time()
    if verbose:
        print(
            f"DRAWING SCAFFOLDS FOR {scaffolds_type} DONE in {end - start:.04} seconds"
        )
    ####################################
