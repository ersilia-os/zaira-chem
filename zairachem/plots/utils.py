import seaborn as sns
import colorsys


def set_style(style=None):
    """Set basic plotting style and fonts."""
    if style is None:
        style = (
            "ticks",
            {
                "font.family": "sans-serif",
                "font.serif": ["Arial"],
                "font.size": 16,
                "axes.grid": True,
            },
        )
    else:
        style = style
    sns.set_style(*style)


def rgb2hex(r, g, b):
    """RGB to hexadecimal."""
    return "#%02x%02x%02x" % (r, g, b)


def predefined_cc_colors(coord, lighness=0):
    """Predefined CC colors."""
    colors = {
        "A": "#EA5A49",  # '#EE7B6D', '#F7BDB6'],
        "B": "#B16BA8",  # '#C189B9', '#D0A6CB'],
        "C": "#5A72B5",  # '#7B8EC4', '#9CAAD3'],
        "D": "#7CAF2A",  # '#96BF55', '#B0CF7F'],
        "E": "#F39426",  # '#F5A951', '#F8BF7D'],
        "Z": "#000000",  # '#666666', '#999999']
    }
    if not coord in colors:
        coord = "Z"
    return lighten_color(colors[coord[:1]], amount=1 - lighness)


def lighten_color(color, amount=0):
    if amount == 0:
        return color
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
