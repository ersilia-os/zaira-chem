import matplotlib
import seaborn as sns

sns.set_style("ticks")

matplotlib.rc("font", family="sans-serif")
matplotlib.rc("font", serif="Arial")
matplotlib.rc("text", usetex="false")
matplotlib.rcParams["figure.autolayout"] = True
matplotlib.rcParams.update({"font.size": 16})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


ersilia_colors = {
    "dark": "#50285a",
    "gray": "#d2d2d0",
    "mint": "#bee6b4",
    "white": "#ffffff",
    "purple": "#aa96fa",
    "pink": "#dca0dc",
    "yellow": "#fad782",
    "blue": "#8cc8fa",
    "red": "#faa08c",
}


def rgb2hex(r, g, b):
    """RGB to hexadecimal."""
    return "#%02x%02x%02x" % (r, g, b)
