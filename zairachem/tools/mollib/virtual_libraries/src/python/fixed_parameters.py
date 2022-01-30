"""
Parameters defined as constant across all experiments
"""

PROCESSING_FIXED = {"start_char": "G", "end_char": "E", "pad_char": "A"}

INDICES_TOKEN = {
    "0": "H",
    "1": "9",
    "2": "D",
    "3": "r",
    "4": "T",
    "5": "R",
    "6": "V",
    "7": "4",
    "8": "c",
    "9": "l",
    "10": "b",
    "11": ".",
    "12": "C",
    "13": "Y",
    "14": "s",
    "15": "B",
    "16": "k",
    "17": "+",
    "18": "p",
    "19": "2",
    "20": "7",
    "21": "8",
    "22": "O",
    "23": "%",
    "24": "o",
    "25": "6",
    "26": "N",
    "27": "A",
    "28": "t",
    "29": "$",
    "30": "(",
    "31": "u",
    "32": "Z",
    "33": "#",
    "34": "M",
    "35": "P",
    "36": "G",
    "37": "I",
    "38": "=",
    "39": "-",
    "40": "X",
    "41": "@",
    "42": "E",
    "43": ":",
    "44": "\\",
    "45": ")",
    "46": "i",
    "47": "K",
    "48": "/",
    "49": "{",
    "50": "h",
    "51": "L",
    "52": "n",
    "53": "U",
    "54": "[",
    "55": "0",
    "56": "y",
    "57": "e",
    "58": "3",
    "59": "g",
    "60": "f",
    "61": "}",
    "62": "1",
    "63": "d",
    "64": "W",
    "65": "5",
    "66": "S",
    "67": "F",
    "68": "]",
    "69": "a",
    "70": "m",
}
TOKEN_INDICES = {v: k for k, v in INDICES_TOKEN.items()}


PAPER_FONT = {"tick_font_sz": 15, "label_font_sz": 18, "legend_sz": 16, "title_sz": 22}

# Number of molecules to use to
# make the UMAP plot.
UMAP_PLOT = {"n_dataset": 1000, "n_gen": 1000}

# Number of molecules to use
# to compute the Fréchet distance.
# This is an upper bound.
FRECHET = {"n_data": 5000}

# Color palette for UMAP
COLOR_PAL_CB = {
    "source": "#1575A4",
    "target": "#D55E00",
    "e_start": "#A0E0FF",
    "e_end": "#FFAE6E",
}

DESCRIPTORS = {"names": "(FractionCSP3)"}
